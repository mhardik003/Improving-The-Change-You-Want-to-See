# import torch
# import torch.nn as nn
# from einops import rearrange
# import torch.nn.functional as F

# original

# class CoAttentionModule(nn.Module):
#     def __init__(self, input_channels=2048, hidden_channels=256, attention_type="coam"):
#         super().__init__()
#         if attention_type == "coam":
#             self.attention_layer = CoAttentionLayer(input_channels, hidden_channels)
#         elif attention_type == "noam":
#             self.attention_layer = NoAttentionLayer()
#         else:
#             raise NotImplementedError(f"Unknown attention {attention_type}")

#     def forward(self, left_features, right_features):
#         weighted_r = self.attention_layer(left_features, right_features)
#         weighted_l = self.attention_layer(right_features, left_features)
#         left_attended_features = rearrange(
#             [left_features, weighted_r], "two b c h w -> b (two c) h w"
#         )
#         right_attended_features = rearrange(
#             [right_features, weighted_l], "two b c h w -> b (two c) h w"
#         )
#         return left_attended_features, right_attended_features


# class CoAttentionLayer(nn.Module):
#     def __init__(self, input_channels=2048, hidden_channels=256):
#         super().__init__()
#         self.reference_dimensionality_reduction = nn.Conv2d(
#             input_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
#         )
#         self.query_dimensionality_reduction = nn.Conv2d(
#             input_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
#         )

#     def forward(self, query_features, reference_features):
#         Q = self.query_dimensionality_reduction(query_features)
#         K = self.reference_dimensionality_reduction(reference_features)
#         V = rearrange(reference_features, "b c h w -> b c (h w)")
#         attention_map = torch.einsum("bcij,bckl->bijkl", Q, K)
#         attention_map = rearrange(attention_map, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)")
#         attention_map = nn.Softmax(dim=3)(attention_map)
#         attended_features = torch.einsum("bijp,bcp->bcij", attention_map, V)
#         return attended_features

# class NoAttentionLayer(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, query_features, reference_features):
#         return reference_features
# original ends
# changed begins

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# class CoAttentionLayer(nn.Module):
#     def __init__(self, input_channels=2048, hidden_channels=256, num_heads=8, max_position=100):
#         super().__init__()
#         self.num_heads = num_heads
#         self.hidden_channels = hidden_channels
#         self.max_position = max_position
#         self.reference_dimensionality_reduction = nn.Conv2d(
#             input_channels, hidden_channels * num_heads, kernel_size=1, stride=1, padding=0, bias=True
#         )
#         self.query_dimensionality_reduction = nn.Conv2d(
#             input_channels, hidden_channels * num_heads, kernel_size=1, stride=1, padding=0, bias=True
#         )
#         self.relative_position_embedding = nn.Parameter(torch.randn(2 * max_position - 1, hidden_channels // num_heads))
#         self.layer_norm = nn.LayerNorm(input_channels)

#     def forward(self, query_features, reference_features, attention_mask=None):
#         Q = self.query_dimensionality_reduction(query_features)
#         K = self.reference_dimensionality_reduction(reference_features)
#         V = rearrange(reference_features, "b c h w -> b c (h w)")

#         Q = rearrange(Q, "b (c h) i j -> b h c i j", h=self.num_heads)
#         K = rearrange(K, "b (c h) i j -> b h c i j", h=self.num_heads)

#         # Compute relative position embeddings
#         query_height, query_width = query_features.shape[2], query_features.shape[3]
#         ref_height, ref_width = reference_features.shape[2], reference_features.shape[3]
#         relative_coords_h = torch.arange(-(ref_height - 1), query_height)
#         relative_coords_w = torch.arange(-(ref_width - 1), query_width)
#         relative_coords = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#         relative_coords = relative_coords.view(-1, 2)
#         relative_position_index = relative_coords[:, 0] * (2 * self.max_position - 1) + relative_coords[:, 1]
#         relative_position_bias = self.relative_position_embedding[relative_position_index.long()]
#         relative_position_bias = relative_position_bias.view(query_height + ref_height - 1, query_width + ref_width - 1, -1)
#         relative_position_bias = F.pad(relative_position_bias, (0, 0, 0, query_height - 1, 0, query_width - 1))
#         relative_position_bias = relative_position_bias[:query_height, :query_width, :]
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0).repeat(self.num_heads, 1, 1, 1)

#         # Compute attention scores with relative position bias
#         attention_map = torch.einsum("bhcij,bhckl->bhijkl", Q, K)
#         print(attention_map.shape)
#         print(relative_position_bias.shape)
#         attention_map = attention_map + relative_position_bias.unsqueeze(-2).unsqueeze(-2)
#         attention_map = rearrange(attention_map, "b h i j k l -> b h i j (k l)")
#         attention_map = attention_map / (self.hidden_channels // self.num_heads ** 0.5)

#         if attention_mask is not None:
#             attention_map = attention_map.masked_fill(attention_mask == 0, float("-inf"))

#         attention_map = nn.Softmax(dim=4)(attention_map)
#         attended_features = torch.einsum("bhijp,bcp->bhcij", attention_map, V)
#         attended_features = rearrange(attended_features, "b h c i j -> b (h c) i j")

#         query_features = self.layer_norm(query_features)
#         attended_features = attended_features + query_features

#         return attended_features

# class NoAttentionLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
# 
#     def forward(self, query_features, reference_features):
#         return reference_features

# changed ends
# final begins

import torch
import torch.nn as nn
from einops import rearrange

class CoAttentionModule(nn.Module):
    def __init__(self, input_channels=2048, hidden_channels=256, num_heads=8, attention_type="coam"):
        super().__init__()
        if attention_type == "coam":
            self.attention_layer = CoAttentionLayer(input_channels, hidden_channels, num_heads)
        elif attention_type == "noam":
            self.attention_layer = NoAttentionLayer()
        else:
            raise NotImplementedError(f"Unknown attention {attention_type}")

    def forward(self, left_features, right_features):
        weighted_r = self.attention_layer(left_features, right_features)
        weighted_l = self.attention_layer(right_features, left_features)

        left_attended_features = rearrange(
            [left_features, weighted_r], "two b c h w -> b (two c) h w"
        )
        right_attended_features = rearrange(
            [right_features, weighted_l], "two b c h w -> b (two c) h w"
        )

        return left_attended_features, right_attended_features

class CoAttentionLayer(nn.Module):
    def __init__(self, input_channels=2048, hidden_channels=256, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

        self.query_projection = nn.Linear(input_channels, hidden_channels * num_heads)
        self.key_projection = nn.Linear(input_channels, hidden_channels * num_heads)
        self.value_projection = nn.Linear(input_channels, hidden_channels * num_heads)
        self.output_projection = nn.Linear(hidden_channels * num_heads, input_channels)

    def forward(self, query_features, reference_features):
        # Determine the device of the query tensor
        query_device = query_features.device

        # Ensure the input tensors are on the same device as the query tensor
        query_features = query_features.to(query_device)
        reference_features = reference_features.to(query_device)

        batch_size, _, height, width = query_features.shape

        query = self.query_projection(rearrange(query_features, "b c h w -> b (h w) c"))
        key = self.key_projection(rearrange(reference_features, "b c h w -> b (h w) c"))
        value = self.value_projection(rearrange(reference_features, "b c h w -> b (h w) c"))

        query = rearrange(query, "b (h w) (head c) -> b head h w c", head=self.num_heads, h=height, w=width)
        key = rearrange(key, "b (h w) (head c) -> b head h w c", head=self.num_heads, h=height, w=width)
        value = rearrange(value, "b (h w) (head c) -> b head h w c", head=self.num_heads, h=height, w=width)

        attention_scores = torch.einsum("bnhwc,bnkwc->bnhwk", query, key)

        relative_indices_h = torch.arange(height).view(1, -1, 1) - torch.arange(height).view(-1, 1, 1)
        relative_indices_w = torch.arange(width).view(1, -1, 1) - torch.arange(width).view(-1, 1, 1)
        relative_indices_h = relative_indices_h.to(query_device)  # Move to the same device as query
        relative_indices_w = relative_indices_w.to(query_device)  # Move to the same device as query

        self.relative_position_embedding = nn.Parameter(torch.randn(2 * max(height, width) - 1, self.hidden_channels).to(query_device))  # Move to the same device as query
        relative_embeddings_h = self.relative_position_embedding[relative_indices_h.view(-1)].view(height, height, 1, self.hidden_channels).to(query_device)  # Move to the same device as query
        relative_embeddings_w = self.relative_position_embedding[relative_indices_w.view(-1)].view(width, width, 1, self.hidden_channels).to(query_device)  # Move to the same device as query

        attention_scores_h = torch.einsum("bnhwc,hkic->bnhwk", query, relative_embeddings_h)
        attention_scores_w = torch.einsum("bnhwc,wkic->bnhwk", query, relative_embeddings_w)
        attention_scores = attention_scores + attention_scores_h + attention_scores_w

        attention_scores = attention_scores / (self.hidden_channels ** 0.5)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attended_features = torch.einsum("bnhwk,bnkwc->bnhwc", attention_probs, value)
        attended_features = rearrange(attended_features, "b head h w c -> b (h w) (head c)")
        attended_features = self.output_projection(attended_features)
        attended_features = rearrange(attended_features, "b (h w) c -> b c h w", h=height, w=width)

        return attended_features


class NoAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_features, reference_features):
        return reference_features