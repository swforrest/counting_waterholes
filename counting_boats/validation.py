import os
import utils.validation as val_utils
import argparse


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.description = "Cluster classifications and labels from yolo and manual annotations to compare them. File names must be identical between given arguments"
    argparser.add_argument("-f", "--folder", help="Directory of classifications and labels", required=True)
    args = argparser.parse_args()
    classifications = os.path.join(args.folder, "Classifications")
    labels = os.path.join(args.folder, "Labels")
    outdir = os.path.join(args.folder, "Summary")
    imgs = os.path.join(args.folder, "SegmentedImages")
    summaries = os.path.join(args.folder, "Summary") 
    # make args.folder global
    folder = args.folder

    # check what we want to do
    print("1. Prepare tif images for labelling")
    print("2. Prepare images for validation")
    print("3. Infer From Images")
    print("4. Analyse Results")
    choice = input("Enter a number: ")
    if choice == "1":
        val_utils.prepare_png_from_tifs(folder)
    elif choice == "2":
        val_utils.prepare_pngs_for_detection(folder)
    elif choice == "3":
        val_utils.run_detection(folder)
    elif choice == "4":
        # First do the comparisons
        val_utils.compare_detections_to_ground_truth(folder)
        # Then summarise the results
        val_utils.summarize(folder)
        # Then plot the boats
        val_utils.plot_boats(summaries, folder)
        # Then highlight the mistakes
        val_utils.highlight_mistakes(folder)




