import torch


def convert_weak_perspective_to_perspective(
    weak_perspective_camera,
    focal_length=5000.,
    img_res=224,
):
    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz]
    # in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    # if isinstance(focal_length, torch.Tensor):
    #     focal_length = focal_length[:, 0]

    perspective_camera = torch.stack([
        weak_perspective_camera[:, 1], weak_perspective_camera[:, 2],
        2 * focal_length / (img_res * weak_perspective_camera[:, 0] + 1e-9)
    ],
                                     dim=-1)
    return perspective_camera


def perspective_projection(points, rotation, translation, focal_length,
                           camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def convert_crop_coords_to_orig_img(bbox, keypoints, crop_size):
    # import IPython; IPython.embed(); exit()
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]

    # unnormalize to crop coords
    keypoints = 0.5 * crop_size * (keypoints + 1.0)

    # rescale to orig img crop
    keypoints *= h[..., None, None] / crop_size

    # transform into original image coords
    keypoints[:, :, 0] = (cx - h / 2)[..., None] + keypoints[:, :, 0]
    keypoints[:, :, 1] = (cy - h / 2)[..., None] + keypoints[:, :, 1]
    return keypoints



if __name__ == '__main__':


            # get smplh vertices from pose & shape
            pred_pose = body3d_result['pred_pose']
            pred_pose = Rot.from_matrix(pred_pose.reshape(
                -1, 3, 3)).as_rotvec().reshape(-1, 24, 3).astype(np.float32)
            pred_pose = torch.from_numpy(
                np.concatenate([
                    pred_pose[:, :22, :],
                    np.zeros((1, 30, 3), dtype=np.float32)
                ],
                               axis=1).reshape(-1, 156))
            pred_shape = body3d_result['pred_shape']
            pred_shape = torch.from_numpy(
                np.concatenate(
                    [pred_shape,
                     np.zeros((1, 6), dtype=np.float32)], axis=1))
            smplh_result = smplh(pred_pose, pred_shape)
            body3d_result['smpl_vertices'] = smplh_result['verts'].cpu().numpy(
            )

            # get hand bbox
            verts_lhand = body3d_result['smpl_vertices'][:, SMPL_LEFT_HAND_IDX]
            verts_rhand = body3d_result['smpl_vertices'][:,
                                                         SMPL_RIGHT_HAND_IDX]

            verts_lhand = torch.from_numpy(verts_lhand).to(device)
            cam_t = convert_weak_perspective_to_perspective(
                torch.Tensor(body3d_result['pred_cam'])).to(device)
            verts2d_lhand = perspective_projection(
                verts_lhand,
                rotation=torch.eye(3, device=device).unsqueeze(0).expand(
                    1, -1, -1),
                translation=cam_t,
                focal_length=5000,
                camera_center=torch.zeros(
                    1, 2, device=device)).cpu().numpy() / (224 / 2.)
            verts2d_lhand_origin = convert_crop_coords_to_orig_img(
                detection_result, verts2d_lhand, 224)

            xmin_lhand, ymin_lhand = np.min(verts2d_lhand_origin[0],
                                            axis=0).astype(np.int32)
            xmax_lhand, ymax_lhand = np.max(verts2d_lhand_origin[0],
                                            axis=0).astype(np.int32)
            bbox_lhand = np.array([[
                xmin_lhand - 10, ymin_lhand - 10, xmax_lhand + 10,
                ymax_lhand + 10
            ]])

            img_lhand = image[ymin_lhand:ymax_lhand, xmin_lhand:xmax_lhand, :]

            verts_rhand = torch.from_numpy(verts_rhand).to(device)
            cam_t = convert_weak_perspective_to_perspective(
                torch.Tensor(body3d_result['pred_cam'])).to(device)
            verts2d_rhand = perspective_projection(
                verts_rhand,
                rotation=torch.eye(3, device=device).unsqueeze(0).expand(
                    1, -1, -1),
                translation=cam_t,
                focal_length=5000,
                camera_center=torch.zeros(
                    1, 2, device=device)).cpu().numpy() / (224 / 2.)
            verts2d_rhand_origin = convert_crop_coords_to_orig_img(
                detection_result, verts2d_rhand, 224)

            xmin_rhand, ymin_rhand = np.min(verts2d_rhand_origin[0],
                                            axis=0).astype(np.int32)
            xmax_rhand, ymax_rhand = np.max(verts2d_rhand_origin[0],
                                            axis=0).astype(np.int32)
            bbox_rhand = np.array([[
                xmin_rhand - 10, ymin_rhand - 10, xmax_rhand + 10,
                ymax_rhand + 10
            ]])

            img_rhand = image[ymin_rhand:ymax_rhand, xmin_rhand:xmax_rhand, :]