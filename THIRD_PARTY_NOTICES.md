# Third-party code

The paper implementation builds on Gaze-LLE and DINOv2. Vendored Python sources retain their original copyright notices and licenses.

| Component | Source revision | License |
| --- | --- | --- |
| [Gaze-LLE](https://github.com/fkryan/gazelle) | `2d78f9f3bd2a5db360354954ec1a4c526ee8ab55` | [MIT](third_party/gazelle/LICENSE) |
| [YuNet face detector (2023mar ONNX)](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet) | `48b10fa505aeb51fbd127fc60a5c660040e823b7` | [MIT](third_party/yunet/LICENSE); model and upstream README bundled; [provenance](third_party/yunet/provenance.json) |
| [DINOv2](https://github.com/facebookresearch/dinov2) | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | [Apache 2.0](third_party/dinov2/LICENSE); additional upstream licenses are retained for Cell DINO and XRay-DINO components |

Only DINOv2 ViT-L/14 is used by the released gaze checkpoint. Its frozen pretrained weights are downloaded from the upstream source or supplied separately through `DINOV2_WEIGHTS`.

The demonstration uses a short clip from VideoAttentionTarget (MLB Interview). The underlying video remains the property of its original rights holders. The clip is included for research demonstration within this private repository.

No additional license is granted here for project-specific code or model weights. This repository is a private research release.
