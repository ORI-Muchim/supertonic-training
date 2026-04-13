
## duration_predictor.onnx  (98 real params, 342,530 elements)

### grouped by prefix depth=3

| prefix | #tensors | numel | example |
|---|---:|---:|---|
| `tts.dp.sentence_encoder` | 93 | 317,696 | `tts.dp.sentence_encoder.text_embedder.char_embedder.weight [163, 64]` |
| `tts.dp.predictor` | 5 | 24,834 | `tts.dp.predictor.layers.0.weight [128, 192]` |

### grouped by prefix depth=4

| prefix | #tensors | numel | example |
|---|---:|---:|---|
| `tts.dp.sentence_encoder.convnext` | 54 | 201,984 | `tts.dp.sentence_encoder.convnext.convnext.0.dwconv.weight [64, 1, 5]` |
| `tts.dp.sentence_encoder.attn_encoder` | 36 | 101,120 | `tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_q.weight [64, 64, 1]` |
| `tts.dp.predictor.layers` | 4 | 24,833 | `tts.dp.predictor.layers.0.weight [128, 192]` |
| `tts.dp.sentence_encoder.text_embedder` | 1 | 10,432 | `tts.dp.sentence_encoder.text_embedder.char_embedder.weight [163, 64]` |
| `tts.dp.sentence_encoder.proj_out` | 1 | 4,096 | `tts.dp.sentence_encoder.proj_out.net.weight [64, 64, 1]` |
| `tts.dp.sentence_encoder.sentence_token` | 1 | 64 | `tts.dp.sentence_encoder.sentence_token [1, 64, 1]` |
| `tts.dp.predictor.activation` | 1 | 1 | `tts.dp.predictor.activation.weight [1]` |

### grouped by prefix depth=5

| prefix | #tensors | numel | example |
|---|---:|---:|---|
| `tts.dp.sentence_encoder.convnext.convnext` | 54 | 201,984 | `tts.dp.sentence_encoder.convnext.convnext.0.dwconv.weight [64, 1, 5]` |
| `tts.dp.sentence_encoder.attn_encoder.ffn_layers` | 8 | 66,176 | `tts.dp.sentence_encoder.attn_encoder.ffn_layers.0.conv_1.weight [256, 64, 1]` |
| `tts.dp.sentence_encoder.attn_encoder.attn_layers` | 20 | 34,432 | `tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_q.weight [64, 64, 1]` |
| `tts.dp.predictor.layers.0` | 2 | 24,704 | `tts.dp.predictor.layers.0.weight [128, 192]` |
| `tts.dp.sentence_encoder.text_embedder.char_embedder` | 1 | 10,432 | `tts.dp.sentence_encoder.text_embedder.char_embedder.weight [163, 64]` |
| `tts.dp.sentence_encoder.proj_out.net` | 1 | 4,096 | `tts.dp.sentence_encoder.proj_out.net.weight [64, 64, 1]` |
| `tts.dp.sentence_encoder.attn_encoder.norm_layers_1` | 4 | 256 | `tts.dp.sentence_encoder.attn_encoder.norm_layers_1.0.norm.weight [64]` |
| `tts.dp.sentence_encoder.attn_encoder.norm_layers_2` | 4 | 256 | `tts.dp.sentence_encoder.attn_encoder.norm_layers_2.0.norm.weight [64]` |
| `tts.dp.predictor.layers.1` | 2 | 129 | `tts.dp.predictor.layers.1.weight [1, 128]` |
| `tts.dp.sentence_encoder.sentence_token` | 1 | 64 | `tts.dp.sentence_encoder.sentence_token [1, 64, 1]` |
| `tts.dp.predictor.activation.weight` | 1 | 1 | `tts.dp.predictor.activation.weight [1]` |

top-level roots: ['tts']

## text_encoder.onnx  (141 real params, 6,767,872 elements)

### grouped by prefix depth=3

| prefix | #tensors | numel | example |
|---|---:|---:|---|
| `tts.ttl.text_encoder` | 127 | 6,372,608 | `tts.ttl.text_encoder.text_embedder.char_embedder.weight [163, 256]` |
| `onnx::MatMul_3680` | 1 | 65,536 | `onnx::MatMul_3680 [256, 256]` |
| `onnx::MatMul_3684` | 1 | 65,536 | `onnx::MatMul_3684 [256, 256]` |
| `onnx::MatMul_3678` | 1 | 65,536 | `onnx::MatMul_3678 [256, 256]` |
| `onnx::MatMul_3681` | 1 | 65,536 | `onnx::MatMul_3681 [256, 256]` |
| `onnx::MatMul_3682` | 1 | 65,536 | `onnx::MatMul_3682 [256, 256]` |
| `onnx::MatMul_3685` | 1 | 65,536 | `onnx::MatMul_3685 [256, 256]` |
| `tts.ttl.speech_prompted_text_encoder` | 8 | 2,048 | `tts.ttl.speech_prompted_text_encoder.attention1.W_value.linear.bias [256]` |

### grouped by prefix depth=4

| prefix | #tensors | numel | example |
|---|---:|---:|---|
| `tts.ttl.text_encoder.convnext` | 54 | 3,167,232 | `tts.ttl.text_encoder.convnext.convnext.0.dwconv.weight [256, 1, 5]` |
| `tts.ttl.text_encoder.attn_encoder` | 72 | 3,163,648 | `tts.ttl.text_encoder.attn_encoder.attn_layers.0.conv_q.weight [256, 256, 1]` |
| `onnx::MatMul_3680` | 1 | 65,536 | `onnx::MatMul_3680 [256, 256]` |
| `onnx::MatMul_3684` | 1 | 65,536 | `onnx::MatMul_3684 [256, 256]` |
| `onnx::MatMul_3678` | 1 | 65,536 | `onnx::MatMul_3678 [256, 256]` |
| `onnx::MatMul_3681` | 1 | 65,536 | `onnx::MatMul_3681 [256, 256]` |
| `onnx::MatMul_3682` | 1 | 65,536 | `onnx::MatMul_3682 [256, 256]` |
| `onnx::MatMul_3685` | 1 | 65,536 | `onnx::MatMul_3685 [256, 256]` |
| `tts.ttl.text_encoder.text_embedder` | 1 | 41,728 | `tts.ttl.text_encoder.text_embedder.char_embedder.weight [163, 256]` |
| `tts.ttl.speech_prompted_text_encoder.attention1` | 3 | 768 | `tts.ttl.speech_prompted_text_encoder.attention1.W_value.linear.bias [256]` |
| `tts.ttl.speech_prompted_text_encoder.attention2` | 3 | 768 | `tts.ttl.speech_prompted_text_encoder.attention2.W_value.linear.bias [256]` |
| `tts.ttl.speech_prompted_text_encoder.norm` | 2 | 512 | `tts.ttl.speech_prompted_text_encoder.norm.norm.weight [256]` |

### grouped by prefix depth=5

| prefix | #tensors | numel | example |
|---|---:|---:|---|
| `tts.ttl.text_encoder.convnext.convnext` | 54 | 3,167,232 | `tts.ttl.text_encoder.convnext.convnext.0.dwconv.weight [256, 1, 5]` |
| `tts.ttl.text_encoder.attn_encoder.ffn_layers` | 16 | 2,102,272 | `tts.ttl.text_encoder.attn_encoder.ffn_layers.0.conv_1.weight [1024, 256, 1]` |
| `tts.ttl.text_encoder.attn_encoder.attn_layers` | 40 | 1,057,280 | `tts.ttl.text_encoder.attn_encoder.attn_layers.0.conv_q.weight [256, 256, 1]` |
| `onnx::MatMul_3680` | 1 | 65,536 | `onnx::MatMul_3680 [256, 256]` |
| `onnx::MatMul_3684` | 1 | 65,536 | `onnx::MatMul_3684 [256, 256]` |
| `onnx::MatMul_3678` | 1 | 65,536 | `onnx::MatMul_3678 [256, 256]` |
| `onnx::MatMul_3681` | 1 | 65,536 | `onnx::MatMul_3681 [256, 256]` |
| `onnx::MatMul_3682` | 1 | 65,536 | `onnx::MatMul_3682 [256, 256]` |
| `onnx::MatMul_3685` | 1 | 65,536 | `onnx::MatMul_3685 [256, 256]` |
| `tts.ttl.text_encoder.text_embedder.char_embedder` | 1 | 41,728 | `tts.ttl.text_encoder.text_embedder.char_embedder.weight [163, 256]` |
| `tts.ttl.text_encoder.attn_encoder.norm_layers_1` | 8 | 2,048 | `tts.ttl.text_encoder.attn_encoder.norm_layers_1.0.norm.weight [256]` |
| `tts.ttl.text_encoder.attn_encoder.norm_layers_2` | 8 | 2,048 | `tts.ttl.text_encoder.attn_encoder.norm_layers_2.0.norm.weight [256]` |
| `tts.ttl.speech_prompted_text_encoder.norm.norm` | 2 | 512 | `tts.ttl.speech_prompted_text_encoder.norm.norm.weight [256]` |
| `tts.ttl.speech_prompted_text_encoder.attention1.W_value` | 1 | 256 | `tts.ttl.speech_prompted_text_encoder.attention1.W_value.linear.bias [256]` |
| `tts.ttl.speech_prompted_text_encoder.attention2.W_value` | 1 | 256 | `tts.ttl.speech_prompted_text_encoder.attention2.W_value.linear.bias [256]` |
| `tts.ttl.speech_prompted_text_encoder.attention1.W_query` | 1 | 256 | `tts.ttl.speech_prompted_text_encoder.attention1.W_query.linear.bias [256]` |
| `tts.ttl.speech_prompted_text_encoder.attention1.out_fc` | 1 | 256 | `tts.ttl.speech_prompted_text_encoder.attention1.out_fc.linear.bias [256]` |
| `tts.ttl.speech_prompted_text_encoder.attention2.W_query` | 1 | 256 | `tts.ttl.speech_prompted_text_encoder.attention2.W_query.linear.bias [256]` |
| `tts.ttl.speech_prompted_text_encoder.attention2.out_fc` | 1 | 256 | `tts.ttl.speech_prompted_text_encoder.attention2.out_fc.linear.bias [256]` |

top-level roots: ['onnx::MatMul_3678', 'onnx::MatMul_3680', 'onnx::MatMul_3681', 'onnx::MatMul_3682', 'onnx::MatMul_3684', 'onnx::MatMul_3685', 'tts']

## vector_estimator.onnx  (349 real params, 33,011,018 elements)

### grouped by prefix depth=3

| prefix | #tensors | numel | example |
|---|---:|---:|---|
| `tts.ttl.vector_field` | 312 | 29,734,216 | `tts.ttl.vector_field.proj_in.net.weight [512, 144, 1]` |
| `onnx::MatMul_3101` | 1 | 131,072 | `onnx::MatMul_3101 [512, 256]` |
| `onnx::MatMul_3110` | 1 | 131,072 | `onnx::MatMul_3110 [256, 512]` |
| `onnx::MatMul_3116` | 1 | 131,072 | `onnx::MatMul_3116 [512, 256]` |
| `onnx::MatMul_3119` | 1 | 131,072 | `onnx::MatMul_3119 [256, 512]` |
| `onnx::MatMul_3146` | 1 | 131,072 | `onnx::MatMul_3146 [512, 256]` |
| `onnx::MatMul_3155` | 1 | 131,072 | `onnx::MatMul_3155 [256, 512]` |
| `onnx::MatMul_3161` | 1 | 131,072 | `onnx::MatMul_3161 [512, 256]` |
| `onnx::MatMul_3164` | 1 | 131,072 | `onnx::MatMul_3164 [256, 512]` |
| `onnx::MatMul_3191` | 1 | 131,072 | `onnx::MatMul_3191 [512, 256]` |
| `onnx::MatMul_3200` | 1 | 131,072 | `onnx::MatMul_3200 [256, 512]` |
| `onnx::MatMul_3206` | 1 | 131,072 | `onnx::MatMul_3206 [512, 256]` |
| `onnx::MatMul_3209` | 1 | 131,072 | `onnx::MatMul_3209 [256, 512]` |
| `onnx::MatMul_3236` | 1 | 131,072 | `onnx::MatMul_3236 [512, 256]` |
| `onnx::MatMul_3245` | 1 | 131,072 | `onnx::MatMul_3245 [256, 512]` |
| `onnx::MatMul_3251` | 1 | 131,072 | `onnx::MatMul_3251 [512, 256]` |
| `onnx::MatMul_3254` | 1 | 131,072 | `onnx::MatMul_3254 [256, 512]` |
| `onnx::MatMul_3118` | 1 | 65,536 | `onnx::MatMul_3118 [256, 256]` |
| `onnx::MatMul_3163` | 1 | 65,536 | `onnx::MatMul_3163 [256, 256]` |
| `onnx::MatMul_3208` | 1 | 65,536 | `onnx::MatMul_3208 [256, 256]` |
| `onnx::MatMul_3253` | 1 | 65,536 | `onnx::MatMul_3253 [256, 256]` |
| `onnx::MatMul_3102` | 1 | 65,536 | `onnx::MatMul_3102 [256, 256]` |
| `onnx::MatMul_3103` | 1 | 65,536 | `onnx::MatMul_3103 [256, 256]` |
| `onnx::MatMul_3147` | 1 | 65,536 | `onnx::MatMul_3147 [256, 256]` |
| `onnx::MatMul_3148` | 1 | 65,536 | `onnx::MatMul_3148 [256, 256]` |
| `onnx::MatMul_3192` | 1 | 65,536 | `onnx::MatMul_3192 [256, 256]` |
| `onnx::MatMul_3193` | 1 | 65,536 | `onnx::MatMul_3193 [256, 256]` |
| `onnx::MatMul_3237` | 1 | 65,536 | `onnx::MatMul_3237 [256, 256]` |
| `onnx::MatMul_3238` | 1 | 65,536 | `onnx::MatMul_3238 [256, 256]` |
| `onnx::MatMul_3117` | 1 | 65,536 | `onnx::MatMul_3117 [256, 256]` |
| `onnx::MatMul_3162` | 1 | 65,536 | `onnx::MatMul_3162 [256, 256]` |
| `onnx::MatMul_3207` | 1 | 65,536 | `onnx::MatMul_3207 [256, 256]` |
| `onnx::MatMul_3252` | 1 | 65,536 | `onnx::MatMul_3252 [256, 256]` |
| `onnx::MatMul_3095` | 1 | 32,768 | `onnx::MatMul_3095 [64, 512]` |
| `onnx::MatMul_3140` | 1 | 32,768 | `onnx::MatMul_3140 [64, 512]` |
| `onnx::MatMul_3185` | 1 | 32,768 | `onnx::MatMul_3185 [64, 512]` |
| `onnx::MatMul_3230` | 1 | 32,768 | `onnx::MatMul_3230 [64, 512]` |
| `onnx::ReduceSum_1279` | 1 | 2 | `onnx::ReduceSum_1279 [2]` |

### grouped by prefix depth=4

| prefix | #tensors | numel | example |
|---|---:|---:|---|
| `tts.ttl.vector_field.main_blocks` | 270 | 25,334,792 | `tts.ttl.vector_field.main_blocks.5.attention.W_value.linear.bias [256]` |
| `tts.ttl.vector_field.last_convnext` | 36 | 4,218,880 | `tts.ttl.vector_field.last_convnext.convnext.0.dwconv.weight [512, 1, 5]` |
| `onnx::MatMul_3101` | 1 | 131,072 | `onnx::MatMul_3101 [512, 256]` |
| `onnx::MatMul_3110` | 1 | 131,072 | `onnx::MatMul_3110 [256, 512]` |
| `onnx::MatMul_3116` | 1 | 131,072 | `onnx::MatMul_3116 [512, 256]` |
| `onnx::MatMul_3119` | 1 | 131,072 | `onnx::MatMul_3119 [256, 512]` |
| `onnx::MatMul_3146` | 1 | 131,072 | `onnx::MatMul_3146 [512, 256]` |
| `onnx::MatMul_3155` | 1 | 131,072 | `onnx::MatMul_3155 [256, 512]` |
| `onnx::MatMul_3161` | 1 | 131,072 | `onnx::MatMul_3161 [512, 256]` |
| `onnx::MatMul_3164` | 1 | 131,072 | `onnx::MatMul_3164 [256, 512]` |
| `onnx::MatMul_3191` | 1 | 131,072 | `onnx::MatMul_3191 [512, 256]` |
| `onnx::MatMul_3200` | 1 | 131,072 | `onnx::MatMul_3200 [256, 512]` |
| `onnx::MatMul_3206` | 1 | 131,072 | `onnx::MatMul_3206 [512, 256]` |
| `onnx::MatMul_3209` | 1 | 131,072 | `onnx::MatMul_3209 [256, 512]` |
| `onnx::MatMul_3236` | 1 | 131,072 | `onnx::MatMul_3236 [512, 256]` |
| `onnx::MatMul_3245` | 1 | 131,072 | `onnx::MatMul_3245 [256, 512]` |
| `onnx::MatMul_3251` | 1 | 131,072 | `onnx::MatMul_3251 [512, 256]` |
| `onnx::MatMul_3254` | 1 | 131,072 | `onnx::MatMul_3254 [256, 512]` |
| `tts.ttl.vector_field.proj_in` | 1 | 73,728 | `tts.ttl.vector_field.proj_in.net.weight [512, 144, 1]` |
| `tts.ttl.vector_field.proj_out` | 1 | 73,728 | `tts.ttl.vector_field.proj_out.net.weight [144, 512, 1]` |
| `onnx::MatMul_3118` | 1 | 65,536 | `onnx::MatMul_3118 [256, 256]` |
| `onnx::MatMul_3163` | 1 | 65,536 | `onnx::MatMul_3163 [256, 256]` |
| `onnx::MatMul_3208` | 1 | 65,536 | `onnx::MatMul_3208 [256, 256]` |
| `onnx::MatMul_3253` | 1 | 65,536 | `onnx::MatMul_3253 [256, 256]` |
| `onnx::MatMul_3102` | 1 | 65,536 | `onnx::MatMul_3102 [256, 256]` |
| `onnx::MatMul_3103` | 1 | 65,536 | `onnx::MatMul_3103 [256, 256]` |
| `onnx::MatMul_3147` | 1 | 65,536 | `onnx::MatMul_3147 [256, 256]` |
| `onnx::MatMul_3148` | 1 | 65,536 | `onnx::MatMul_3148 [256, 256]` |
| `onnx::MatMul_3192` | 1 | 65,536 | `onnx::MatMul_3192 [256, 256]` |
| `onnx::MatMul_3193` | 1 | 65,536 | `onnx::MatMul_3193 [256, 256]` |
| `onnx::MatMul_3237` | 1 | 65,536 | `onnx::MatMul_3237 [256, 256]` |
| `onnx::MatMul_3238` | 1 | 65,536 | `onnx::MatMul_3238 [256, 256]` |
| `onnx::MatMul_3117` | 1 | 65,536 | `onnx::MatMul_3117 [256, 256]` |
| `onnx::MatMul_3162` | 1 | 65,536 | `onnx::MatMul_3162 [256, 256]` |
| `onnx::MatMul_3207` | 1 | 65,536 | `onnx::MatMul_3207 [256, 256]` |
| `onnx::MatMul_3252` | 1 | 65,536 | `onnx::MatMul_3252 [256, 256]` |
| `tts.ttl.vector_field.time_encoder` | 4 | 33,088 | `tts.ttl.vector_field.time_encoder.mlp.0.linear.weight [256, 64]` |
| `onnx::MatMul_3095` | 1 | 32,768 | `onnx::MatMul_3095 [64, 512]` |
| `onnx::MatMul_3140` | 1 | 32,768 | `onnx::MatMul_3140 [64, 512]` |
| `onnx::MatMul_3185` | 1 | 32,768 | `onnx::MatMul_3185 [64, 512]` |
| `onnx::MatMul_3230` | 1 | 32,768 | `onnx::MatMul_3230 [64, 512]` |
| `onnx::ReduceSum_1279` | 1 | 2 | `onnx::ReduceSum_1279 [2]` |

### grouped by prefix depth=5

| prefix | #tensors | numel | example |
|---|---:|---:|---|
| `tts.ttl.vector_field.main_blocks.0` | 36 | 4,218,880 | `tts.ttl.vector_field.main_blocks.0.convnext.0.dwconv.weight [512, 1, 5]` |
| `tts.ttl.vector_field.main_blocks.6` | 36 | 4,218,880 | `tts.ttl.vector_field.main_blocks.6.convnext.0.dwconv.weight [512, 1, 5]` |
| `tts.ttl.vector_field.main_blocks.12` | 36 | 4,218,880 | `tts.ttl.vector_field.main_blocks.12.convnext.0.dwconv.weight [512, 1, 5]` |
| `tts.ttl.vector_field.main_blocks.18` | 36 | 4,218,880 | `tts.ttl.vector_field.main_blocks.18.convnext.0.dwconv.weight [512, 1, 5]` |
| `tts.ttl.vector_field.last_convnext.convnext` | 36 | 4,218,880 | `tts.ttl.vector_field.last_convnext.convnext.0.dwconv.weight [512, 1, 5]` |
| `tts.ttl.vector_field.main_blocks.2` | 9 | 1,054,720 | `tts.ttl.vector_field.main_blocks.2.convnext.0.dwconv.weight [512, 1, 5]` |
| `tts.ttl.vector_field.main_blocks.4` | 9 | 1,054,720 | `tts.ttl.vector_field.main_blocks.4.convnext.0.dwconv.weight [512, 1, 5]` |
| `tts.ttl.vector_field.main_blocks.8` | 9 | 1,054,720 | `tts.ttl.vector_field.main_blocks.8.convnext.0.dwconv.weight [512, 1, 5]` |
| `tts.ttl.vector_field.main_blocks.10` | 9 | 1,054,720 | `tts.ttl.vector_field.main_blocks.10.convnext.0.dwconv.weight [512, 1, 5]` |
| `tts.ttl.vector_field.main_blocks.14` | 9 | 1,054,720 | `tts.ttl.vector_field.main_blocks.14.convnext.0.dwconv.weight [512, 1, 5]` |
| `tts.ttl.vector_field.main_blocks.16` | 9 | 1,054,720 | `tts.ttl.vector_field.main_blocks.16.convnext.0.dwconv.weight [512, 1, 5]` |
| `tts.ttl.vector_field.main_blocks.20` | 9 | 1,054,720 | `tts.ttl.vector_field.main_blocks.20.convnext.0.dwconv.weight [512, 1, 5]` |
| `tts.ttl.vector_field.main_blocks.22` | 9 | 1,054,720 | `tts.ttl.vector_field.main_blocks.22.convnext.0.dwconv.weight [512, 1, 5]` |
| `onnx::MatMul_3101` | 1 | 131,072 | `onnx::MatMul_3101 [512, 256]` |
| `onnx::MatMul_3110` | 1 | 131,072 | `onnx::MatMul_3110 [256, 512]` |
| `onnx::MatMul_3116` | 1 | 131,072 | `onnx::MatMul_3116 [512, 256]` |
| `onnx::MatMul_3119` | 1 | 131,072 | `onnx::MatMul_3119 [256, 512]` |
| `onnx::MatMul_3146` | 1 | 131,072 | `onnx::MatMul_3146 [512, 256]` |
| `onnx::MatMul_3155` | 1 | 131,072 | `onnx::MatMul_3155 [256, 512]` |
| `onnx::MatMul_3161` | 1 | 131,072 | `onnx::MatMul_3161 [512, 256]` |
| `onnx::MatMul_3164` | 1 | 131,072 | `onnx::MatMul_3164 [256, 512]` |
| `onnx::MatMul_3191` | 1 | 131,072 | `onnx::MatMul_3191 [512, 256]` |
| `onnx::MatMul_3200` | 1 | 131,072 | `onnx::MatMul_3200 [256, 512]` |
| `onnx::MatMul_3206` | 1 | 131,072 | `onnx::MatMul_3206 [512, 256]` |
| `onnx::MatMul_3209` | 1 | 131,072 | `onnx::MatMul_3209 [256, 512]` |
| `onnx::MatMul_3236` | 1 | 131,072 | `onnx::MatMul_3236 [512, 256]` |
| `onnx::MatMul_3245` | 1 | 131,072 | `onnx::MatMul_3245 [256, 512]` |
| `onnx::MatMul_3251` | 1 | 131,072 | `onnx::MatMul_3251 [512, 256]` |
| `onnx::MatMul_3254` | 1 | 131,072 | `onnx::MatMul_3254 [256, 512]` |
| `tts.ttl.vector_field.proj_in.net` | 1 | 73,728 | `tts.ttl.vector_field.proj_in.net.weight [512, 144, 1]` |
| `tts.ttl.vector_field.proj_out.net` | 1 | 73,728 | `tts.ttl.vector_field.proj_out.net.weight [144, 512, 1]` |
| `onnx::MatMul_3118` | 1 | 65,536 | `onnx::MatMul_3118 [256, 256]` |
| `onnx::MatMul_3163` | 1 | 65,536 | `onnx::MatMul_3163 [256, 256]` |
| `onnx::MatMul_3208` | 1 | 65,536 | `onnx::MatMul_3208 [256, 256]` |
| `onnx::MatMul_3253` | 1 | 65,536 | `onnx::MatMul_3253 [256, 256]` |
| `onnx::MatMul_3102` | 1 | 65,536 | `onnx::MatMul_3102 [256, 256]` |
| `onnx::MatMul_3103` | 1 | 65,536 | `onnx::MatMul_3103 [256, 256]` |
| `onnx::MatMul_3147` | 1 | 65,536 | `onnx::MatMul_3147 [256, 256]` |
| `onnx::MatMul_3148` | 1 | 65,536 | `onnx::MatMul_3148 [256, 256]` |
| `onnx::MatMul_3192` | 1 | 65,536 | `onnx::MatMul_3192 [256, 256]` |
| `onnx::MatMul_3193` | 1 | 65,536 | `onnx::MatMul_3193 [256, 256]` |
| `onnx::MatMul_3237` | 1 | 65,536 | `onnx::MatMul_3237 [256, 256]` |
| `onnx::MatMul_3238` | 1 | 65,536 | `onnx::MatMul_3238 [256, 256]` |
| `onnx::MatMul_3117` | 1 | 65,536 | `onnx::MatMul_3117 [256, 256]` |
| `onnx::MatMul_3162` | 1 | 65,536 | `onnx::MatMul_3162 [256, 256]` |
| `onnx::MatMul_3207` | 1 | 65,536 | `onnx::MatMul_3207 [256, 256]` |
| `onnx::MatMul_3252` | 1 | 65,536 | `onnx::MatMul_3252 [256, 256]` |
| `tts.ttl.vector_field.time_encoder.mlp` | 4 | 33,088 | `tts.ttl.vector_field.time_encoder.mlp.0.linear.weight [256, 64]` |
| `onnx::MatMul_3095` | 1 | 32,768 | `onnx::MatMul_3095 [64, 512]` |
| `onnx::MatMul_3140` | 1 | 32,768 | `onnx::MatMul_3140 [64, 512]` |
| `onnx::MatMul_3185` | 1 | 32,768 | `onnx::MatMul_3185 [64, 512]` |
| `onnx::MatMul_3230` | 1 | 32,768 | `onnx::MatMul_3230 [64, 512]` |
| `tts.ttl.vector_field.main_blocks.3` | 8 | 3,336 | `tts.ttl.vector_field.main_blocks.3.attn.W_key.linear.bias [256]` |
| `tts.ttl.vector_field.main_blocks.5` | 6 | 2,304 | `tts.ttl.vector_field.main_blocks.5.attention.W_value.linear.bias [256]` |
| `tts.ttl.vector_field.main_blocks.11` | 6 | 2,304 | `tts.ttl.vector_field.main_blocks.11.attention.W_value.linear.bias [256]` |
| `tts.ttl.vector_field.main_blocks.17` | 6 | 2,304 | `tts.ttl.vector_field.main_blocks.17.attention.W_value.linear.bias [256]` |
| `tts.ttl.vector_field.main_blocks.23` | 6 | 2,304 | `tts.ttl.vector_field.main_blocks.23.attention.W_value.linear.bias [256]` |
| `tts.ttl.vector_field.main_blocks.9` | 6 | 2,304 | `tts.ttl.vector_field.main_blocks.9.attn.W_key.linear.bias [256]` |
| `tts.ttl.vector_field.main_blocks.15` | 6 | 2,304 | `tts.ttl.vector_field.main_blocks.15.attn.W_key.linear.bias [256]` |
| `tts.ttl.vector_field.main_blocks.21` | 6 | 2,304 | `tts.ttl.vector_field.main_blocks.21.attn.W_key.linear.bias [256]` |
| `tts.ttl.vector_field.main_blocks.1` | 1 | 512 | `tts.ttl.vector_field.main_blocks.1.linear.linear.bias [512]` |
| `tts.ttl.vector_field.main_blocks.7` | 1 | 512 | `tts.ttl.vector_field.main_blocks.7.linear.linear.bias [512]` |
| `tts.ttl.vector_field.main_blocks.13` | 1 | 512 | `tts.ttl.vector_field.main_blocks.13.linear.linear.bias [512]` |
| `tts.ttl.vector_field.main_blocks.19` | 1 | 512 | `tts.ttl.vector_field.main_blocks.19.linear.linear.bias [512]` |
| `onnx::ReduceSum_1279` | 1 | 2 | `onnx::ReduceSum_1279 [2]` |

top-level roots: ['onnx::MatMul_3095', 'onnx::MatMul_3101', 'onnx::MatMul_3102', 'onnx::MatMul_3103', 'onnx::MatMul_3110', 'onnx::MatMul_3116', 'onnx::MatMul_3117', 'onnx::MatMul_3118', 'onnx::MatMul_3119', 'onnx::MatMul_3140', 'onnx::MatMul_3146', 'onnx::MatMul_3147', 'onnx::MatMul_3148', 'onnx::MatMul_3155', 'onnx::MatMul_3161', 'onnx::MatMul_3162', 'onnx::MatMul_3163', 'onnx::MatMul_3164', 'onnx::MatMul_3185', 'onnx::MatMul_3191', 'onnx::MatMul_3192', 'onnx::MatMul_3193', 'onnx::MatMul_3200', 'onnx::MatMul_3206', 'onnx::MatMul_3207', 'onnx::MatMul_3208', 'onnx::MatMul_3209', 'onnx::MatMul_3230', 'onnx::MatMul_3236', 'onnx::MatMul_3237', 'onnx::MatMul_3238', 'onnx::MatMul_3245', 'onnx::MatMul_3251', 'onnx::MatMul_3252', 'onnx::MatMul_3253', 'onnx::MatMul_3254', 'onnx::ReduceSum_1279', 'tts']

## vocoder.onnx  (103 real params, 25,338,418 elements)

### grouped by prefix depth=3

| prefix | #tensors | numel | example |
|---|---:|---:|---|
| `tts.ae.decoder` | 97 | 25,251,840 | `tts.ae.decoder.convnext.0.dwconv.net.weight [512, 1, 7]` |
| `onnx::Conv_1440` | 1 | 86,016 | `onnx::Conv_1440 [512, 24, 7]` |
| `onnx::Conv_1441` | 1 | 512 | `onnx::Conv_1441 [512]` |
| `tts.ae.latent_std` | 1 | 24 | `tts.ae.latent_std [1, 24, 1]` |
| `tts.ae.latent_mean` | 1 | 24 | `tts.ae.latent_mean [1, 24, 1]` |
| `tts.ttl.normalizer` | 1 | 1 | `tts.ttl.normalizer.scale []` |
| `onnx::PRelu_1505` | 1 | 1 | `onnx::PRelu_1505 [1, 1]` |

### grouped by prefix depth=4

| prefix | #tensors | numel | example |
|---|---:|---:|---|
| `tts.ae.decoder.convnext` | 90 | 21,053,440 | `tts.ae.decoder.convnext.0.dwconv.net.weight [512, 1, 7]` |
| `tts.ae.decoder.head` | 3 | 4,196,352 | `tts.ae.decoder.head.layer1.net.weight [2048, 512, 3]` |
| `onnx::Conv_1440` | 1 | 86,016 | `onnx::Conv_1440 [512, 24, 7]` |
| `tts.ae.decoder.final_norm` | 4 | 2,048 | `tts.ae.decoder.final_norm.norm.weight [512]` |
| `onnx::Conv_1441` | 1 | 512 | `onnx::Conv_1441 [512]` |
| `tts.ae.latent_std` | 1 | 24 | `tts.ae.latent_std [1, 24, 1]` |
| `tts.ae.latent_mean` | 1 | 24 | `tts.ae.latent_mean [1, 24, 1]` |
| `tts.ttl.normalizer.scale` | 1 | 1 | `tts.ttl.normalizer.scale []` |
| `onnx::PRelu_1505` | 1 | 1 | `onnx::PRelu_1505 [1, 1]` |

### grouped by prefix depth=5

| prefix | #tensors | numel | example |
|---|---:|---:|---|
| `tts.ae.decoder.head.layer1` | 2 | 3,147,776 | `tts.ae.decoder.head.layer1.net.weight [2048, 512, 3]` |
| `tts.ae.decoder.convnext.0` | 9 | 2,105,344 | `tts.ae.decoder.convnext.0.dwconv.net.weight [512, 1, 7]` |
| `tts.ae.decoder.convnext.1` | 9 | 2,105,344 | `tts.ae.decoder.convnext.1.dwconv.net.weight [512, 1, 7]` |
| `tts.ae.decoder.convnext.2` | 9 | 2,105,344 | `tts.ae.decoder.convnext.2.dwconv.net.weight [512, 1, 7]` |
| `tts.ae.decoder.convnext.3` | 9 | 2,105,344 | `tts.ae.decoder.convnext.3.dwconv.net.weight [512, 1, 7]` |
| `tts.ae.decoder.convnext.4` | 9 | 2,105,344 | `tts.ae.decoder.convnext.4.dwconv.net.weight [512, 1, 7]` |
| `tts.ae.decoder.convnext.5` | 9 | 2,105,344 | `tts.ae.decoder.convnext.5.dwconv.net.weight [512, 1, 7]` |
| `tts.ae.decoder.convnext.6` | 9 | 2,105,344 | `tts.ae.decoder.convnext.6.dwconv.net.weight [512, 1, 7]` |
| `tts.ae.decoder.convnext.7` | 9 | 2,105,344 | `tts.ae.decoder.convnext.7.dwconv.net.weight [512, 1, 7]` |
| `tts.ae.decoder.convnext.8` | 9 | 2,105,344 | `tts.ae.decoder.convnext.8.dwconv.net.weight [512, 1, 7]` |
| `tts.ae.decoder.convnext.9` | 9 | 2,105,344 | `tts.ae.decoder.convnext.9.dwconv.net.weight [512, 1, 7]` |
| `tts.ae.decoder.head.layer2` | 1 | 1,048,576 | `tts.ae.decoder.head.layer2.weight [512, 2048, 1]` |
| `onnx::Conv_1440` | 1 | 86,016 | `onnx::Conv_1440 [512, 24, 7]` |
| `tts.ae.decoder.final_norm.norm` | 4 | 2,048 | `tts.ae.decoder.final_norm.norm.weight [512]` |
| `onnx::Conv_1441` | 1 | 512 | `onnx::Conv_1441 [512]` |
| `tts.ae.latent_std` | 1 | 24 | `tts.ae.latent_std [1, 24, 1]` |
| `tts.ae.latent_mean` | 1 | 24 | `tts.ae.latent_mean [1, 24, 1]` |
| `tts.ttl.normalizer.scale` | 1 | 1 | `tts.ttl.normalizer.scale []` |
| `onnx::PRelu_1505` | 1 | 1 | `onnx::PRelu_1505 [1, 1]` |

top-level roots: ['onnx::Conv_1440', 'onnx::Conv_1441', 'onnx::PRelu_1505', 'tts']
