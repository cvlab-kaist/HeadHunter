import os
import torch
from ..utils import logging
import warnings
from pprint import pprint
import numpy as np
from ..models.attention_processor import (
    JointAttnProcessor2_0,
    SaveJointAttnProcessor2_0,
)
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class AttnMixin:
    r"""Mixin class for saving attention map."""

    attention_maps_list = []
    attention_values_list = []
    attn_output_list = []
    gate_msa_list = []
    attn_hidden_states_list = []
    gated_attn_output_list = []
    hidden_states_list = []

    num_layers = 0
    timestep = 0
    
    def checkTransformer(self):
        return hasattr(self, "transformer")
    
    def check_attn_processor(self, name, module):
        class_name = self.__class__.__name__
        cond = False
        if (class_name == 'StableDiffusion3Pipeline') and hasattr(module, "processor") and name.endswith("attn"):
            cond = True
        elif (class_name == 'CogVideoXPipeline') and hasattr(module, "processor") and name.endswith("attn"):
            cond = True
        return cond
    
    def save_transformer_block(self, save_gate_msa=False, save_attn_hidden_states=False, save_gated_attn_output=False, save_hidden_states=False):
        for block in self.transformer.transformer_blocks:
            block.register_forward_hook(self.save_transformer_block_hook(save_gate_msa, save_attn_hidden_states, save_gated_attn_output, save_hidden_states))

    def save_transformer_block_hook(self, save_gate_msa=False, save_attn_hidden_states=False, save_gated_attn_output=False, save_hidden_states=False):
        def save_transformer_block_hook_(module, input, output):
            if save_gate_msa:
                self.gate_msa_list.append(output[-1].detach().cpu().numpy().astype(np.float16))
            if save_attn_hidden_states:
                self.attn_hidden_states_list.append(output[-2].detach().cpu().numpy().astype(np.float16))
            if save_gated_attn_output:
                self.gated_attn_output_list.append(output[-3].detach().cpu().numpy().astype(np.float16))
            if save_hidden_states:
                self.hidden_states_list.append(output[-4].clone().detach().cpu().numpy().astype(np.float16))
            return output[:2]
        return save_transformer_block_hook_

    def set_attention_maps_saving(self, save_values=False, save_attn_output=False, save_attn_heads=False, save_attn_dir = "./"):
            r"""Set attention processors to save_attention_maps."""

            if self.checkTransformer():
                self.num_layers = len(self.transformer.transformer_blocks)
                print(f"Number of layers: {self.num_layers}")

                for name, module in self.transformer.named_modules():
                    if self.check_attn_processor(name, module) and not isinstance(module.processor, SaveJointAttnProcessor2_0):
                        if isinstance(module.processor, JointAttnProcessor2_0):
                            module.processor = SaveJointAttnProcessor2_0()
                        else :
                            raise ValueError(f"Attention processor {module.processor.__class__.__name__} is not supported.")
                        module.register_forward_hook(
                            self.save(name,  save_values=save_values, save_attn_output=save_attn_output, save_attn_heads=save_attn_heads, save_attn_dir=save_attn_dir))

            else:
                raise ValueError("Invalid model type")
            
    def save(self, layer_name, save_values=False, save_attn_output=False, save_attn_heads=False, save_attn_dir="./"):
        def save_attention_maps(module, input, output):
            r"""Save attention maps."""
            attn_weight = output[-2] # .detach().cpu().numpy().astype(np.float16), # (batch_size, attn.heads, seq_len, seq_len)
            if save_attn_heads:
                attn_weight = attn_weight.squeeze().detach().cpu().numpy().astype(np.float16) # (attn.heads, seq_len, seq_len)
            else:
                attn_weight = (
                    torch.mean(attn_weight, dim=1).squeeze().detach().cpu().numpy().astype(np.float16)
                )  # Average over heads, torch.Size([1, 12, 4096, 4360])
            self.attention_maps_list.append(attn_weight)

            # Save Norm of Value
            if save_values:
                value = output[-1]  # (batch_size, attn.heads, seq_len, head_dim)
                value = value.transpose(1,2) # (batch_size, seq_len, attn.heads, head_dim)
                value = value.reshape(value.shape[0], value.shape[1], -1) # (batch_size, seq_len, attn.heads*head_dim)
                value = value.detach().cpu().numpy().astype(np.float16)
                # value = torch.norm(value, dim=-1)[0].detach().cpu().numpy()
                self.attention_values_list.append(value)

            # Save Attention Output
            if save_attn_output:
                attn_output = output[0]
                attn_output = attn_output.detach().cpu().numpy().astype(np.float16)
                self.attn_output_list.append(attn_output)


            # Save Attention for every num_layers
            if len(self.attention_maps_list) == self.num_layers:
                if save_attn_heads :
                    attention_maps = np.array(self.attention_maps_list)
                    filename = os.path.join(save_attn_dir, f"timestep{self.timestep}_attention_maps_heads.npy")
                    np.save(filename, attention_maps)
                    self.attention_maps_list = []
                    print(f"timestep{self.timestep} attention_maps_heads saved")
                self.timestep += 1
                

            # Return remaining of tuple except for attn_weight
            if len(output) >= 4:
                return output[:-2]
            else :
                return output[0]

        return save_attention_maps