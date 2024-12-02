import torch


def video_data_collate_fn(batch, frames_per_clip=1, frame_interval=1):
    collated_batch = {}
    for key in batch[0].keys():
        if key == "video" or key == "video_latent":
            data_to_stack = []
            for data in batch:
                # here we randomly sample a clip from the video, len=MAX_SEQ_LEN
                MAX_SEQ_LEN = len(data[key])

                if MAX_SEQ_LEN == frames_per_clip:
                    start_idx = 0
                    end_idx = MAX_SEQ_LEN
                else:
                    # start_idx = torch.randint(0, MAX_SEQ_LEN - frames_per_clip, (1,))
                    # we need to take fomr start_idx to start_idx + frames_per_clip*frame_interval

                    if not (MAX_SEQ_LEN > frames_per_clip * frame_interval + 1):
                        frame_interval = max(1, MAX_SEQ_LEN // frames_per_clip)

                    start_idx = torch.randint(
                        0, MAX_SEQ_LEN - frames_per_clip * frame_interval, (1,)
                    )
                    end_idx = start_idx + frames_per_clip * frame_interval

                if not isinstance(data[key], torch.Tensor):
                    data_to_stack += [
                        torch.cat(
                            data[key][start_idx:end_idx:frame_interval],
                            dim=0,
                        )
                    ]
                else:
                    data_to_stack += [data[key][start_idx:end_idx:frame_interval]]

            # we return image anyhow
            collated_batch[key.replace("video", "image")] = torch.stack(data_to_stack)
        else:
            if not isinstance(batch[0][key], torch.Tensor):
                collated_batch[key] = torch.stack(
                    [torch.tensor(data[key]) for data in batch]
                )
            else:
                collated_batch[key] = torch.stack([data[key] for data in batch])

    return collated_batch
