For 3DADN, refer to the [original repository](https://github.com/JasonQSY/Articulation3D) for setup instructions.

For our experiments, we used the pretrained model provided by 3DADN, which is trained on their Internet video dataset.

To run our experiments, use the following command
```
cd articulation3d/
python tools/inference.py --config/config.yaml  --input /path/to/videos/list --out_path /path/to/output/folder --save-obj --webvis
```