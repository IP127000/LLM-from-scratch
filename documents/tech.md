一、RMSNorm操作
代码实现几乎一样，但在返回值的操作上略有不同：
OSS: 
    return (self.weight * hidden_states).to(input_dtype) 
Qwen3:
    return self.weight * hidden_states.to(input_dtype) 
更推荐OSS的方式

二、MoE层
1.专家网络权重形式不一样
OSS采用大矩阵存储所有的MLP,且有bias，gate_up_proj形状为num_experts*hidden_size*(2*expert_dim),合并了两个上采样的权重；
down_proj的权重形状为：num_experts*expert_dim*hidden_size

Qwen3采用存储的粒度更细，每个专家的权重独立存储，且门控和上采样权重分开，不使用bias,形状均为hidden_size*expert_dim;
下采样权重为：expert_dim*hiddent_size
所有专家的权重定义为：nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

2.专家的前向计算方式不一样
Qwen3采用：Loop over all available experts
OSS可以有两种方式：
 When training is is more efficient to just loop over the experts and compute the output for each expert
For inference we can sacrifice some memory and compute the output for all experts at once. 

3.激活函数
OSS使用：
    glu = gate * torch.sigmoid(gate * self.alpha) 
Qwen3使用：
    silu
4.MLP结构不一样，OSS including clamping and a residual connection
OSS使用：
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu
                out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]

Qwen使用：
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

三、RoPE代码一样
四、 eager_attention的计算
    OSS使用了GQA和sink attention
            combined_logits = torch.cat([attn_weights, sinks], dim=-1)
            combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
            probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
            scores = probs[..., :-1]
    Qwen3使用了GQA，未使用sink attention
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
五、 Attention类型不一样
    OSS交替使用sliding_attention,full_attention
    Qwen3支持sliding_window
六、负载均衡
    均使用Switch Transformer
