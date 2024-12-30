import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

import numpy as np

def partition_batch(batch, num_partitions, partition_width, halo_width):
    data = []
    filter = [np.zeros(partition_width+2*halo_width, dtype=bool) for _ in range(num_partitions)]

    data.append(batch[:, :, 0:partition_width+2*halo_width, :, :])
    for i in range(1, num_partitions-1):
        data.append(batch[:, :, i*partition_width-halo_width:(i+1)*partition_width+halo_width, :, :])
    data.append(batch[:, :, (num_partitions-1)*partition_width-2*halo_width:, :, :])

    # Create filter for inner nodes
    filter[0][0:partition_width] = 1
    for i in range(1, num_partitions-1):
        filter[i][halo_width:partition_width+halo_width] = 1
    filter[num_partitions-1][2*halo_width:] = 1

    return data, filter

def partition_full(batch, num_partitions, partition_width):

    data = []
    total_node = num_partitions*partition_width
    filter = [np.zeros(total_node, dtype=bool) for _ in range(num_partitions)]

    data.append(batch[:, :, :, :, :])
    for i in range(1, num_partitions-1):
        data.append(batch[:, :, :, :, :])
    data.append(batch[:, :, :, :, :])

    # Create filter for inner nodes
    filter[0][0:partition_width] = 1
    for i in range(1, num_partitions-1):
        filter[i][partition_width*i:partition_width*(i+1)] = 1
    filter[num_partitions-1][total_node-partition_width:] = 1

    return data, filter


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, intermediate_channels=128, out_channels=1, kernel_size=2, num_layers=7, padding='same'):
        super().__init__()
        latent_dim = intermediate_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        # Embedding MLP to map from input channels to intermediate channels
        self.embedding_mlp = nn.Sequential(
            nn.Linear(in_channels, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)  # Added LayerNorm after embedding MLP
        )

        # Convolutional layers and MLPs for intermediate processing
        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.layer_norms = nn.ModuleList()  # Added ModuleList for LayerNorms

        # Initial convolution layer (after embedding MLP)
        self.convs.append(nn.Conv3d(latent_dim, intermediate_channels, kernel_size, padding=padding, bias=True))
        self.mlps.append(self._build_mlp(intermediate_channels, latent_dim))
        self.layer_norms.append(nn.LayerNorm(latent_dim))

        # Intermediate layers (intermediate to intermediate)
        for _ in range(num_layers - 2):
            self.convs.append(nn.Conv3d(intermediate_channels, intermediate_channels, kernel_size, padding=padding, bias=True))
            self.mlps.append(self._build_mlp(intermediate_channels, latent_dim))
            self.layer_norms.append(nn.LayerNorm(latent_dim))

        # Final convolution layer (intermediate to output)
        self.convs.append(nn.Conv3d(intermediate_channels, out_channels, kernel_size, padding=padding, bias=True))

        # MLP for the final layer
        self.mlps.append(self._build_mlp(out_channels, out_channels))
        self.layer_norms.append(nn.Identity())

    def _build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, output_dim)
        )

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()

        # Apply embedding MLP to map from input channels to intermediate channels
        x = x.view(batch_size, channels, depth, height, width)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (batch_size, depth, height, width, channels)
        x = x.view(-1, channels)  # Flatten to (batch_size * depth * height * width, channels)
        x = self.embedding_mlp(x)
        x = x.view(batch_size, depth, height, width, self.latent_dim).permute(0, 4, 1, 2, 3).contiguous()  # (batch_size, latent_dim, depth, height, width)

        # Pass through each convolutional layer followed by the pixel-wise MLP and LayerNorm
        for i, (conv, mlp, layer_norm) in enumerate(zip(self.convs, self.mlps, self.layer_norms)):
            x_in = x
            x = conv(x)  # Apply convolution
            x = F.relu(x)  # Apply ReLU

            # Reshape to (batch_size * depth * height * width, channels) for pixel-wise MLP
            x = x.permute(0, 2, 3, 4, 1).contiguous()  # (batch_size, depth, height, width, channels)
            if i < len(self.convs) - 1:  # For all but the last layer, use latent_dim
                x = x.view(-1, self.latent_dim)  # Flatten to (batch_size * depth * height * width, latent_dim)
            else:  # For the last layer, map to out_channels
                x = x.view(-1, self.out_channels)  # Flatten to (batch_size * depth * height * width, out_channels)

            # Apply MLP to each voxel's channels
            x = mlp(x)

            # Apply LayerNorm
            x = layer_norm(x)

            # Reshape back to (batch_size, channels, depth, height, width)
            if i < len(self.convs) - 1:  # For all but the last layer
                x = x.view(batch_size, depth, height, width, self.latent_dim).permute(0, 4, 1, 2, 3).contiguous()
                x = x + x_in
            else:  # For the last layer
                x = x.view(batch_size, depth, height, width, self.out_channels).permute(0, 4, 1, 2, 3).contiguous()

        return x

# Function to train the model on full image or patches
def train_model_base(model, data, target, optimizer):
    model.train()
    optimizer.zero_grad()

    output = model(data)

    loss = torch.mean((output-target)**2)
    loss.backward()
    optimizer.step()

    return output.detach(), loss.item()

# Function to train the model on patches with halo
def train_model_patch(model, data, target, optimizer):
    #model.train()
    optimizer.zero_grad()
    total_loss = 0

    outputs = []
    # Apply halo and split into patches
    patches, filter = partition_batch(data, 10, 100, 10)
    target_patches, _ = partition_batch(target, 10, 100, 10)

    # Process each patch
    for i, patch in enumerate(patches):
        output_patch = model(patch)
        output_patch_filtered = output_patch[:, :, list(filter[i])]
        target_filtered = target_patches[i][:, :, list(filter[i])]
        loss = torch.mean((output_patch_filtered - target_filtered)**2) / len(patches)
        outputs.append(output_patch_filtered)
        total_loss += loss.item()
        loss.backward()
    outputs = torch.cat(outputs, dim=2)

    optimizer.step()

    return outputs, total_loss

# Initialize model, optimizer, and loss function
model_full = SimpleCNN(in_channels=3, out_channels=1, kernel_size=(2,1,1), num_layers=10).to(dtype=torch.float64)
model_patch = copy.deepcopy(model_full)

optimizer_full = optim.Adam(model_full.parameters(), lr=0.001)
optimizer_patch = copy.deepcopy(optimizer_full)

# Generate dummy data
batch_size = 1
in_channels = 3
depth = 1000
height = 1
width = 1

data = torch.randn(batch_size, in_channels, depth, height, width, dtype=torch.float64)  # Random input tensor
target = torch.randn(batch_size, 1, depth, height, width, dtype=torch.float64)  # Random target tensor

# Training loop for comparison
output_full, loss_full = train_model_base(model_full, data, target, optimizer_full)
print(f"Full Image Training - Loss: {loss_full:.4f}")

output_patch, loss_patch = train_model_patch(model_patch, data, target, optimizer_patch)
print(f"Patches Training - Loss: {loss_patch:.4f}")

# Compare outputs and loss
epsilon = 1e-9  # Small value to prevent division by zero
#output_diff = torch.abs(output_full - output_patch).mean().item()

output_rel_error = torch.abs(output_full - output_patch) / (torch.abs(output_full) + epsilon)
top_values, flat_indices = torch.topk(output_rel_error.flatten(), 5)
multi_dim_indices = torch.unravel_index(flat_indices, output_rel_error.shape)


print(f"Output difference: {top_values}")
print(f"Output val: {output_full[multi_dim_indices]}")



# Optionally, compare gradients (if needed)
grad_diff = 0
i = 0
max_diff = []
for p1, p2 in zip(model_full.parameters(), model_patch.parameters()):
    grad_rel_error = torch.abs(p1.grad - p2.grad) / (torch.abs(p1.grad) + epsilon)
    if grad_rel_error.numel() < 5:
        top_values, flat_indices = torch.topk(grad_rel_error.flatten(), grad_rel_error.numel())
    else:
        top_values, flat_indices = torch.topk(grad_rel_error.flatten(), 5)
    multi_dim_indices = torch.unravel_index(flat_indices, grad_rel_error.shape)

    max_diff.append(top_values.detach())

my_tensor = torch.cat(max_diff, dim=0)
top_values, flat_indices = torch.topk(my_tensor.flatten(), 5)

print(f"Grad rel error at layer {top_values}")

