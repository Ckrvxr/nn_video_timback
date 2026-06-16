import torch
import torch.nn.functional as F

def tiled_inference(model, x, tile_size=1024, overlap=64):
    """
    Apply model in tiles to save memory. 
    Assumes model can take (B, C, H, W) and return (B, C, H, W).
    """
    B, C, H, W = x.shape
    output = torch.zeros_like(x)
    
    for y in range(0, H, tile_size - overlap):
        for x_tile in range(0, W, tile_size - overlap):
            y_end = min(y + tile_size, H)
            x_end = min(x_tile + tile_size, W)
            
            # Adjust start to keep tile size constant if possible (except for last tiles)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            tile = x[:, :, y_start:y_end, x_start:x_end]
            tile_out = model(tile)
            
            # Handle overlap (simple crop for now, or feathering)
            # For simplicity, we just write the non-overlapping part or handle edges
            # In a real implementation, we'd use a weight mask for feathering.
            
            # Simple approach: write the central part
            out_y_start = y_start + (overlap // 2 if y_start > 0 else 0)
            out_x_start = x_start + (overlap // 2 if x_start > 0 else 0)
            out_y_end = y_end - (overlap // 2 if y_end < H else 0)
            out_x_end = x_end - (overlap // 2 if x_end < W else 0)
            
            tile_y_start = overlap // 2 if y_start > 0 else 0
            tile_x_start = overlap // 2 if x_start > 0 else 0
            tile_y_end = (y_end - y_start) - (overlap // 2 if y_end < H else 0)
            tile_x_end = (x_end - x_start) - (overlap // 2 if x_end < W else 0)
            
            output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = \
                tile_out[:, :, tile_y_start:tile_y_end, tile_x_start:tile_x_end]
                
    return output
