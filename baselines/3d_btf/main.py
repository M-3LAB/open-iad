import argparse
from patchcore_runner import PatchCore
from data.mvtec3d import mvtec3d_classes
import pandas as pd


def run_3d_ads():
    classes = mvtec3d_classes()
    METHOD_NAMES = [
        "RGB iNet",
        "Depth iNet",
        "Raw",
        "HoG",
        "SIFT",
        "FPFH",
        "RGB + FPFH"]

    image_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    pixel_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    for cls in classes:
        print(f"\nRunning on class {cls}\n")
        patchcore = PatchCore()
        patchcore.fit(cls)
        image_rocaucs, pixel_rocaucs, au_pros = patchcore.evaluate(cls)
        image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map(image_rocaucs)
        pixel_rocaucs_df[cls.title()] = pixel_rocaucs_df['Method'].map(pixel_rocaucs)
        au_pros_df[cls.title()] = au_pros_df['Method'].map(au_pros)

        print(f"\nFinished running on class {cls}")
        print("################################################################################\n\n")

    image_rocaucs_df['Mean'] = round(image_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    pixel_rocaucs_df['Mean'] = round(pixel_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    au_pros_df['Mean'] = round(au_pros_df.iloc[:, 1:].mean(axis=1),3)

    print("\n\n################################################################################")
    print("############################# Image ROCAUC Results #############################")
    print("################################################################################\n")
    print(image_rocaucs_df.to_markdown(index=False))

    print("\n\n################################################################################")
    print("############################# Pixel ROCAUC Results #############################")
    print("################################################################################\n")
    print(pixel_rocaucs_df.to_markdown(index=False))

    print("\n\n##########################################################################")
    print("############################# AU PRO Results #############################")
    print("##########################################################################\n")
    print(au_pros_df.to_markdown(index=False))



    with open("results/image_rocauc_results.md", "w") as tf:
        tf.write(image_rocaucs_df.to_markdown(index=False))
    with open("results/pixel_rocauc_results.md", "w") as tf:
        tf.write(pixel_rocaucs_df.to_markdown(index=False))
    with open("results/aupro_results.md", "w") as tf:
        tf.write(au_pros_df.to_markdown(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    args = parser.parse_args()

    run_3d_ads()
