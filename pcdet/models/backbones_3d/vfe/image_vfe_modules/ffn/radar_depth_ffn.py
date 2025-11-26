from . import ddn, ddn_loss 
import torch

class RadarDepthFFN:
    def __init__(self, model_cfg, downsample_factor, use_lidar_depth, use_pooling, use_depth):
        # Initialize layers and parameters
        self.ddn = ddn.__all__[model_cfg.DDN.NAME](
            num_classes=model_cfg.DISC_CFG["num_bins"] + 1,
            backbone_name=model_cfg.DDN.BACKBONE_NAME,
            use_lidar_depth=use_lidar_depth,
            use_pooling=use_pooling,
            use_depth=use_depth,
            **model_cfg.DDN.ARGS
        )

    def forward(self, batch_dict, rcs_embedding, mlp_input):
        '''
        Predicts depths and create image depth feature volume using depth distributions and augmented radar points 
        Args: 
            batch_dict:
                images: (N, 3, H_in, W_in), Input images 
                radars: (N, M, 5), Augmented radar points with depth information
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features

        '''
        # Pixel-wise depth classification with radar augmentation 
        images = batch_dict["images"] # [2,3,516,1936]
        ddn_result = self.ddn(images) # 이거 대신에 Depth Net사용 
        # ㅇ너란얼나ㅣ얼나ㅓㅣㄴ알
        
    
        pass