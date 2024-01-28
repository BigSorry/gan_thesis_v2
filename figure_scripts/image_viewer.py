import glob
import matplotlib.pyplot as plt
import numpy as np
import re
from PIL import Image
from pathlib import Path

def sortAUCDimensionImages(images, max_columns):
    dict_info = {}
    dimensions = []
    for image in images:
        split_str = image.filename.split("_")
        auc_intervals = split_str[-3]
        dimension = split_str[-1]
        left_interval = float(re.findall(r'\d\.\d+', auc_intervals)[0])
        dimension = int(re.findall(r'\d+', dimension)[0])
        if dimension not in dict_info:
            dimensions.append(dimension)
            dict_info[dimension] = {}
        dict_info[dimension][left_interval] = image
    # Sorting on auc and dimension
    sorted_dims = sorted(dimensions)
    all_sorted = []
    for dim in sorted_dims:
        images_dict = dict_info[dim]
        if len(images_dict) == max_columns:
            sorted_images_dim = [images_dict[key] for key in sorted(images_dict.keys())]
            for image in sorted_images_dim:
                all_sorted.append(image)
        # Not all auc ranges have an image for this dimension

    return all_sorted

def sortDimensionImages(images):
    dict_info = {}
    for image in images:
        split_str = image.filename.split("_")
        dimension = int(re.findall(r'\d+', split_str[-1])[0])
        dict_info[dimension] = image
    sorted_images = [dict_info[key] for key in sorted(dict_info.keys())]

    return sorted_images

def filenameToTitle(filename):
    group_name = filename.rstrip(".png")
    first_digit_index = next((index for index, char in enumerate(group_name) if char.isdigit()), None)
    if first_digit_index is not None:
        # Insert a space after the first digit
        output_string = group_name[:first_digit_index-1] + " [" + group_name[first_digit_index:]
    else:
        print("No digit found in the string.")
        output_string = group_name

    return output_string

def saveCombinedPlot(map_path_images, variable_name, max_columns, save_file):
    images = [Image.open(image) for image in glob.glob(map_path_images+"*.png")]
    if "dim" in variable_name:
        if "auc" in variable_name:
            images = sortAUCDimensionImages(images, max_columns)
        else:
            images = sortDimensionImages(images)
    total_images = len(images)
    rows = np.ceil(total_images / max_columns).astype(int)
    width = 8
    height = 12
    print(rows, max_columns)
    fig, axs = plt.subplots(rows, max_columns,
                            figsize=(width, height))
    axs = axs.flatten()
    for index, image in enumerate(images):
        width, height = image.size
        new = image.resize((width, height))
        resized = np.asarray(new)
        # path\group_name.png assumed structure
        # filename = (image.filename.split("\\"))[-1]
        # subtitle = filenameToTitle(filename)
        # axs[index].set_title(subtitle)

        axs[index].axis("off")
        axs[index].imshow(image)

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_file, bbox_inches="tight", format="pdf", dpi=150)
    plt.close()

def saveMaps(metric_name, scaled_mode, variable_name):
    max_columns = 1
    file_names = f"{variable_name}*.png"
    if variable_name == "dimension":
        max_columns = 2
    scenario_str = f"{metric_name}_{scaled_mode}"
    map_path = f"../test_scripts/boxplot_all_k/filter1.1/{scenario_str}/{variable_name}/"
    save_name = f"../combined_figures/{variable_name}/{scenario_str}_no_filter.png"
    saveCombinedPlot(map_path, file_names, max_columns, save_name)
    map_path = f"../test_scripts/boxplot_all_k/filter0.002/{scenario_str}/{variable_name}/"
    save_name = f"../combined_figures/{variable_name}/{scenario_str}_filter0.002.png"
    saveCombinedPlot(map_path, file_names, max_columns, save_name)

def makeCombinedFigures():
    metrics = ["pr", "dc"]
    scale_modes = ["real_scaled"]
    sub_maps = ["auc", "dimension"]
    for metric in metrics:
        for scale_mode in scale_modes:
            for sub_map in sub_maps:
                saveMaps(metric, scale_mode, sub_map)

def doIterations(max_winners, metrics, scale_modes, variable_name, scource_map, save_map, max_columns=2):
    for max_winner in max_winners:
        for metric_name in metrics:
            for scaling_mode in scale_modes:
                scenario_str = f"{metric_name}_{scaling_mode}"
                source_map_images = f"{scource_map}{variable_name}/{scenario_str}/"
                # boxplot_all_k
                source_map_images = f"{scource_map}{scenario_str}/{variable_name}/"
                print(source_map_images)
                save_file = f"{save_map}{scenario_str}_max_winner{max_winner}.pdf"
                saveCombinedPlot(source_map_images, variable_name, max_columns, save_file)

def combineHeatmaps():
    metrics = ["pr", "dc"]
    scale_modes = ["real_scaled", "fake_scaled"]
    variable_names = ["auc", "dimension"]
    variable_names = ["auc_dimension"]
    # Lines 1 col and heatmap 2
    max_columns = 1
    max_winners = [1]
    for variable_name in variable_names:
        source_map_images = f"../experiment_figures/heatmap_test/max_winner1_pick_best/"
        save_map = f"../combined_figures/heatmap_combined/{variable_name}/"
        Path(save_map).mkdir(parents=True, exist_ok=True)
        doIterations(max_winners, metrics, scale_modes, variable_name,
                     source_map_images, save_map, max_columns=max_columns)

def combineBoxplot():
    metrics = ["pr", "dc"]
    scale_modes = ["real_scaled", "fake_scaled"]
    variable_names = ["auc", "dimension", "auc_dimension"]
    variable_names = ["auc_dimension"]
    # cols AUC , Dim 3, AUC_Dim is 3
    max_columns = 2
    max_columns = 3
    max_winners = [1]
    for variable_name in variable_names:
        source_map_images = f"../test_scripts/boxplot_all_k/max_winner1/"
        save_map = f"../combined_figures/boxplot_combined/{variable_name}/"
        Path(save_map).mkdir(parents=True, exist_ok=True)
        doIterations(max_winners, metrics, scale_modes, variable_name, source_map_images, save_map, max_columns=max_columns)

def combineAUCDimension():
    metrics = ["pr", "dc"]
    scale_modes = ["real_scaled", "fake_scaled"]
    variable_name = "auc_dimension"
    max_columns = 3
    filter_percentages = [0.002]
    quantiles = [0.25, 0.5, 0.75]
    for filter_percentage in filter_percentages:
        for metric_name in metrics:
            for scaling_mode in scale_modes:
                scenario_str = f"{metric_name}_{scaling_mode}"
                for quant in quantiles:
                    save_map = f"../combined_figures/lines_combined/{variable_name}/{scenario_str}_q{quant}/"
                    Path(save_map).mkdir(parents=True, exist_ok=True)
                    save_file = f"{save_map}f{filter_percentage}.pdf"
                    source_map_images = f"../experiment_figures/lines/{variable_name}/{scenario_str}_f{filter_percentage}_q{quant}/"

                    saveCombinedPlot(source_map_images, variable_name, max_columns, save_file)

#combineHeatmaps()
combineBoxplot()
#combineAUCDimension()