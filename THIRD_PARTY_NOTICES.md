# Third-party notices

Glimpse builds on [Context-Aware PoseFormer](https://github.com/QitaoZhao/ContextAware-PoseFormer), by Qitao Zhao and collaborators. Training, dataset and utility code, CPN integration, and Transformer components are derived from that implementation. The upstream repository did not provide a root license file at the time of this release; no new license is asserted over that upstream code.

- `code/mvn/models/pose_hrnet.py` retains Microsoft's copyright and MIT notice. The upstream HRNet MIT license is reproduced in `licenses/HRNet-MIT.txt`.
- `code/mvn/models/pose_glimpse.py` retains its attribution to the timm Vision Transformer implementation. The timm Apache-2.0 license is reproduced in `licenses/timm-Apache-2.0.txt`.
- `code/mvn/models/networks/` and `cpn/` are the CPN backbone integration inherited from Context-Aware PoseFormer. Their original ownership and applicable terms are retained.
- H36M-Toolbox, VideoPose3D detections, Human3.6M data and pretrained backbone files are external resources, not redistributed in this source tree. Follow each provider's terms.

The top-level MIT license applies to original Glimpse contributions only. It does not replace third-party licenses or grant additional rights over upstream components, datasets, or poster imagery.
