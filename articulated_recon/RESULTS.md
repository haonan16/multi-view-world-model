| scene | source | input | robot mm | gripper mm | joint ° | peg mm | target mm | axis ° | travel mm | IoU fit | IoU held-out | depth agree | time s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard | Sim GT (package, 512px, 5 keyframes) | multiview video + depth + cameras | 1.2 | 1.6 | 0.21 | 3.7 | 6.5 | – | – | 0.972 | – | 0.959 | 122 |
| standard | Sim GT (package, 512px, 5 keyframes) | monocular video + depth + camera | 2.8 | 15.3 | 2.84 | 15.4 | 9.7 | – | – | 0.960 | 0.969 | 0.953 | 114 |
| standard | Sim GT (package, 512px, 5 keyframes) | monocular video only (pixels) | 23.3 | 19.1 | 3.50 | 131.3 | 68.4 | – | – | 0.844 | 0.611 | 0.268 | 325 |
| standard | Sim GT (model frames, 448×256, 11 frames) | multiview video + depth + cameras | 1.3 | 1.1 | 0.41 | 3.8 | 6.6 | – | – | – | – | 0.896 | 188 |
| standard | Sim GT (model frames, 448×256, 11 frames) | multiview + depth + cameras, then + pixel silhouettes (3D→pixels hybrid) | 1.3 | 1.8 | 0.46 | 3.8 | 6.6 | – | – | – | – | 0.896 | 188 |
| standard | Sim GT (model frames, 448×256, 11 frames) | monocular video + depth + camera | 165.2 | 147.6 | 22.44 | 16.5 | 270.9 | – | – | – | – | 0.288 | 189 |
| standard | Sim GT (model frames, 448×256, 11 frames) | monocular video only (pixels) | 928.4 | 919.5 | 21.56 | 1114.1 | 905.6 | – | – | – | – | 0.000 | 644 |
| standard | Model B: real RGB → predicted depth | multiview video + depth + cameras | 27.1 | 74.8 | 22.10 | 134.8 | 262.5 | – | – | – | – | 0.171 | 142 |
| standard | Model B: real RGB → predicted depth | multiview + depth + cameras, then + pixel silhouettes (3D→pixels hybrid) | 18.8 | 61.1 | 17.22 | 134.8 | 262.5 | – | – | – | – | 0.153 | 142 |
| standard | Model B: real RGB → predicted depth | multiview + depth + cameras, depth self-calibrated by the robot | 61.9 | 105.2 | 35.21 | 33.4 | 242.5 | – | – | – | – | 0.127 | 152 |
| standard | Model B: real RGB → predicted depth | multiview + depth (self-calibrated), then + pixel silhouettes (hybrid) | 56.0 | 100.8 | 32.69 | 33.4 | 242.5 | – | – | – | – | 0.130 | 152 |
| standard | Model B: real RGB → predicted depth | monocular video + depth + camera | 107.6 | 229.4 | 26.11 | 429.3 | 30.2 | – | – | – | – | 0.116 | 153 |
| standard | Model B: real RGB → predicted depth | monocular + depth + camera, depth self-calibrated by the robot | 106.0 | 229.3 | 27.98 | 436.2 | 28.4 | – | – | – | – | 0.147 | 149 |
| standard | Model B rollout: 1 view → 4 views + depth | multiview video + depth + cameras | 135.2 | 225.8 | 51.03 | – | – | – | – | – | – | 0.102 | 145 |
| standard | Model B rollout: 1 view → 4 views + depth | multiview + depth + cameras, depth self-calibrated by the robot | 110.0 | 246.1 | 45.73 | 157.3 | 37.0 | – | – | – | – | 0.055 | 211 |
| standard | Model B rollout: 1 view → 4 views + depth | monocular video + depth + camera | 155.6 | 307.8 | 42.10 | 220.5 | 132.3 | – | – | – | – | 0.096 | 148 |
| standard | Model B rollout: 1 view → 4 views + depth | monocular + depth + camera, depth self-calibrated by the robot | 131.5 | 311.1 | 35.29 | 303.1 | 110.5 | – | – | – | – | 0.087 | 154 |
| standard | Model B rollout: 1 view → 4 views + depth | monocular video only (pixels) | 845.5 | 893.7 | 34.54 | 1018.2 | 853.2 | – | – | – | – | 0.001 | 576 |
| standard | Model A rollout: 1 view → 4 views + depth | multiview + depth + cameras, depth self-calibrated by the robot | 39.8 | 114.6 | 31.52 | 74.5 | 41.5 | – | – | – | – | 0.102 | 260 |
| standard | Model A rollout: 1 view → 4 views + depth | monocular + depth + camera, depth self-calibrated by the robot | 498.9 | 1052.3 | 70.34 | – | – | – | – | – | – | 0.029 | 144 |
| standard | Model A rollout: 1 view → 4 views + depth | monocular video only (pixels) | 879.5 | 865.8 | 25.81 | 762.8 | 764.2 | – | – | – | – | 0.001 | 508 |
| drawer | Sim GT (package, 512px, 5 keyframes) | multiview video + depth + cameras | 2.1 | 3.3 | 0.65 | – | – | 0.0 | 1.6 | 0.948 | – | 0.834 | 115 |
| drawer | Sim GT (package, 512px, 5 keyframes) | monocular video + depth + camera | 1.7 | 1.9 | 0.49 | – | – | 47.0 | 45.4 | 0.655 | 0.628 | 0.486 | 129 |
| drawer | Sim GT (package, 512px, 5 keyframes) | monocular video only (pixels) | 74.5 | 106.6 | 21.14 | – | – | 126.8 | 490.3 | 0.486 | 0.352 | 0.041 | 223 |
| drawer | Model B: real RGB → predicted depth | multiview video + depth + cameras | 87.4 | 91.1 | 35.49 | – | – | 86.7 | 98.6 | – | – | 0.082 | 91 |
| drawer | Model B: real RGB → predicted depth | multiview + depth + cameras, depth self-calibrated by the robot | 52.0 | 99.1 | 20.38 | – | – | 104.6 | 212.1 | – | – | 0.045 | 102 |
| drawer | Model B: real RGB → predicted depth | monocular video + depth + camera | 83.7 | 87.5 | 23.02 | – | – | 123.0 | 99.1 | – | – | 0.058 | 52 |
| drawer | Model B rollout: 1 view → 4 views + depth | multiview video + depth + cameras | 195.3 | 255.2 | 33.59 | – | – | 159.5 | 76.9 | – | – | 0.084 | 52 |
| drawer | Model B rollout: 1 view → 4 views + depth | multiview + depth + cameras, depth self-calibrated by the robot | 114.7 | 224.5 | 36.11 | – | – | – | – | – | – | 0.034 | 73 |
| drawer | Model B rollout: 1 view → 4 views + depth | monocular video + depth + camera | 233.4 | 364.4 | 37.48 | – | – | 90.7 | 98.1 | – | – | 0.034 | 35 |
| drawer | Model B rollout: 1 view → 4 views + depth | monocular + depth + camera, depth self-calibrated by the robot | 91.7 | 159.5 | 28.45 | – | – | 90.7 | 172.5 | – | – | 0.058 | 86 |
| drawer | Model B rollout: 1 view → 4 views + depth | monocular video only (pixels) | 136.4 | 260.2 | 23.56 | – | – | 133.6 | 396.0 | – | – | 0.026 | 182 |
| real | Real RH20T (5 RealSense views, sensor depth) | multiview video + depth + cameras | 23.5 | 41.6 | 5.03 | – | – | – | – | – | – | 0.275 | 196 |
| real | Real RH20T (5 RealSense views, sensor depth) | multiview + depth + cameras, depth self-calibrated by the robot | 15.0 | 22.5 | 2.73 | – | – | – | – | – | – | 0.448 | 199 |
| real | Real RH20T (5 RealSense views, sensor depth) | multiview + depth (self-calibrated) + cameras re-calibrated on the robot (logged joints, ICP) | 13.0 | 20.9 | 4.21 | – | – | – | – | – | – | 0.505 | 183 |
| real | Real RH20T (5 RealSense views, sensor depth) | monocular + depth + camera, depth self-calibrated by the robot | 15.6 | 23.8 | 5.51 | – | – | – | – | – | – | 0.475 | 144 |
| real | Real RH20T (5 RealSense views, sensor depth) | monocular video only (pixels) | 100.1 | 122.6 | 9.89 | – | – | – | – | – | – | 0.044 | 327 |
