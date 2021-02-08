import nvidia.dali as dali
import nvidia.dali.types as types
import argparse


def main(filename):
    pipe = dali.pipeline.Pipeline(batch_size=3, num_threads=1, device_id=0)
    with pipe:
        images = dali.fn.external_source(device="cpu", name="RAW_IMAGE")
        images = dali.fn.image_decoder(images, device="mixed", output_type=types.RGB)
        images = dali.fn.resize(images, resize_x=300, resize_y=300)
        images = dali.fn.crop_mirror_normalize(images,
                                               dtype=types.FLOAT,
                                               output_layout="CHW",
                                               crop=(300, 300),
                                               mean=[0.0, 0.0, 0.0],
                                               std=[255.0, 255.0, 255.0])
        pipe.set_outputs(images)
        pipe.serialize(filename=filename)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Serialize pipeline and save it to file")
    parser.add_argument('file_path', type=str, help='Path, where to save serialized pipeline')
    args = parser.parse_args()
    main(args.file_path)
