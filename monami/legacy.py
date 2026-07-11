"""Matplotlib-based plotting and legacy helpers for backward compatibility."""

from __future__ import annotations

import random
import time

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy import stats

from monami.io import load_exhaustive_3d as _load_exhaustive_3d
from monami.sampling import stratified_sample, stratified_sample_dataframe
from monami.transform import (
    categorize as _categorize,
    easyformat_to_numpy2d,
    numpy2d_to_dataframe,
    numpy2d_to_easyformat,
)


def Tic():
    return time.time()


def Toc(start, message, precision=2):
    end = time.time() - start
    end = "{:.{}f}".format(end, precision)
    print(message + end)
    return end


def Histogram(array, limits, bins, title, fn_out):
    print("Histogram : saving in " + fn_out)
    plt.hist(array, bins=bins, range=limits, density=True)
    plt.title(title)
    plt.savefig(fn_out)
    print("     Histogram : Done")


def Show_sample(array, v_min, v_max, samples_df, circle_text, circle_size, circle_color, filename):
    print("Start using Show_sample(). Saving plot: " + filename + "...")
    fig, ax = plt.subplots(1)
    ax.set_aspect("equal")
    normalize = mcolors.Normalize(vmin=v_min, vmax=v_max)
    ax.imshow(array, cmap=cm.jet, origin="lower", norm=normalize)
    for i in range(samples_df.shape[0]):
        x_i = samples_df.iloc[[i]]["X"]
        y_i = samples_df.iloc[[i]]["Y"]
        circ = Circle((x_i, y_i), circle_size, fc=circle_color)
        plt.text(
            x_i,
            y_i,
            str(i),
            ha="center",
            va="center",
            family="sans-serif",
            size=circle_text,
            color="white",
        )
        ax.add_patch(circ)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("     Stop using Show_sample().")


def Plot_df_scatter(df_in, value_column, size, x_min, x_max, y_min, y_max, v_min, v_max, title, filename_out):
    normalize = mcolors.Normalize(vmin=v_min, vmax=v_max)
    ax = df_in.plot.scatter(
        x="X",
        y="Y",
        c=value_column,
        colormap="jet",
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
        title=title,
        norm=normalize,
    )
    ax.set_aspect("equal")
    fig = ax.get_figure()
    fig.savefig(filename_out)


def Stratified_random_sampling_2d_np(exhaustive_array, n_h, n_v):
    print("Start using Sample_simulation()...")
    out = stratified_sample(exhaustive_array, n_h, n_v)
    print("     Stop using Sample_simulation().")
    return out


def Stratified_random_sampling_2d_df(exhaustive_array, n_h, n_v):
    print("Start using Sample_simulation()...")
    sample_out = stratified_sample_dataframe(exhaustive_array, n_h, n_v)
    sample_out = sample_out.rename(columns={"V": "VALUE"})
    print("     Stop using Sample_simulation().")
    return sample_out


def Numpyeasy_to_df(numpy_in):
    return numpy2d_to_dataframe(numpy_in)


def Numpy2d_to_easyformat(numpy_in):
    return numpy2d_to_easyformat(numpy_in)


def Easyformat_to_numpy2d(numpy_in):
    return easyformat_to_numpy2d(numpy_in)


def Categorized(one_d_numpy, rows, columns, q_categories):
    return _categorize(one_d_numpy, rows, columns, q_categories)


def Exhaustive(exhaustive_file, lines_to_jump):
    text_file = open(exhaustive_file, "r")
    print("Reading file :" + exhaustive_file)
    lines = text_file.readlines()[lines_to_jump:]
    text_file.close()
    print("File read    :" + exhaustive_file)
    return np.asarray(lines, dtype=np.float32)


def Show_array(array, v_min, v_max, title, save, filename):
    normalize = mcolors.Normalize(vmin=v_min, vmax=v_max)
    plt.imshow(array, cmap=cm.jet, origin="lower", norm=normalize)
    print("     Start using Show_array() to visualize array...")
    plt.title(title)
    plt.tight_layout()
    plt.colorbar()
    if save == "y":
        plt.savefig(filename)
        print("     Saving plot as file: " + filename)
    print("     Stop using Show_array()")


def Array_exhaustive_2d(exhaustive_file, lines_to_jump, exhaustive_rows, exhaustive_columns):
    text_file = open(exhaustive_file, "r")
    print("Reading file :" + exhaustive_file)
    lines = text_file.readlines()[lines_to_jump:]
    text_file.close()
    print("File read    :" + exhaustive_file)
    lines_np = np.asarray(lines, dtype=np.float32)
    print("Casting array to numpy array")
    lines_2d = np.reshape(lines_np, (exhaustive_rows, exhaustive_columns))
    print("Reshaping array to: " + str(exhaustive_rows) + " rows and: " + str(exhaustive_columns) + " columns")
    return lines_2d


def Array_exhaustive_3d(exhaustive_file, lines_to_jump, rows, columns, levels):
    print("     Reading file :" + exhaustive_file)
    volume = _load_exhaustive_3d(exhaustive_file, lines_to_jump, rows, columns, levels)
    print("     Reshaping array to: " + str(rows) + " rows, " + str(columns) + " columns, and " + str(levels) + " levels")
    return volume


def Write_sgems(df_in, title, spacer, fn_out):
    f = open(fn_out, "w+", newline="")
    f.write(title + "\n")
    f.write("4\n")
    f.write("x\n")
    f.write("y\n")
    f.write("z\n")
    f.write("v\n")
    df_in.to_csv(f, index=False, sep=spacer, header=False)
    f.close()


def Plot_accu_trained_categorical(history, hyper_num, nodes_layer_str, hyper_str, folder):
    acc_key = "acc" if "acc" in history.history else "accuracy"
    val_acc_key = "val_acc" if "val_acc" in history.history else "val_accuracy"
    plt.plot(history.history[acc_key])
    plt.plot(history.history[val_acc_key])
    title = hyper_str[3] + "\n"
    title = title + "Model Accuracy" + "\n"
    title = title + "batch_size:" + hyper_num[1] + "\n"
    title = title + "epochs:" + hyper_num[2] + "\n"
    title = title + "hidden_activation:" + hyper_str[2] + "\n"
    title = title + "loss_function:" + hyper_str[0] + "\n"
    title = title + "optimizer:" + hyper_str[1] + "\n"
    title = title + "test_training_ratio :" + hyper_num[0] + "\n"
    title = title + "dropout:" + hyper_num[3] + "\n"
    nodes = "nodes:"
    for i in nodes_layer_str:
        nodes = nodes + i + ","
    title = title + nodes[:-1] + "\n"
    title = title + "training time [s] :" + hyper_num[4]
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.tight_layout()
    filename_out = (
        hyper_str[4]
        + "_"
        + hyper_num[0]
        + "_"
        + hyper_num[1]
        + "_"
        + hyper_num[2]
        + "_"
        + hyper_num[3]
        + "_"
        + hyper_str[2]
        + "_"
        + hyper_str[0]
        + "_"
        + hyper_str[1]
        + "_"
        + hyper_str[5]
    )
    fn_out = "_"
    for i in nodes_layer_str:
        fn_out = fn_out + i + "_"
    fn_out = filename_out + fn_out[:-1]
    plt.savefig(folder + fn_out + "_accu.pdf")
    plt.close()
    return fn_out


# Remaining legacy geostatistics helpers (unchanged from original workflow)

def Points_to_blocks(filename_in, rows_jump, sep, x_size, y_size, z_size, filename_out_1, filename_out_2):
    print("Reading file: " + filename_in)
    points = pd.read_csv(filename_in, sep=sep, skiprows=rows_jump, header=None, names=["X", "Y", "Z", "V"])
    print("File read.")
    x_min = points["X"].min()
    y_min = points["Y"].min()
    z_min = points["Z"].min()
    points["X"] = points["X"] - x_min
    points["Y"] = points["Y"] - y_min
    points["Z"] = points["Z"] - z_min
    blocks = points.copy()
    blocks["X"] = round(blocks["X"] / x_size + 1).astype(int)
    blocks["Y"] = round(blocks["Y"] / y_size + 1).astype(int)
    blocks["Z"] = round(blocks["Z"] / z_size + 1).astype(int)
    blocks["ID"] = blocks.apply(lambda row: str(row.X) + str(row.Y) + str(row.Z), axis=1)
    blocks["DUP"] = blocks["ID"].duplicated(keep=False)
    blocks["DUP"] = blocks["DUP"].astype(int)
    blocks = blocks.sort_values(["Z", "Y", "X"])
    blocks = blocks.reset_index(drop=True)
    blocks_unique_before = list(blocks["ID"].unique())
    blocks_unique = pd.DataFrame()
    counter = 1
    for iden in blocks_unique_before:
        perc = counter / len(blocks_unique_before) * 100
        prec = 2
        print(
            "Working on dh: "
            + str(counter)
            + " out of :"
            + str(len(blocks_unique_before))
            + ", "
            + "{:.{}f}".format(perc, prec)
            + "%"
        )
        counter = counter + 1
        block_id = blocks[blocks["ID"] == iden]
        block_id = block_id.reset_index(drop=True)
        if block_id["DUP"][0] == 0:
            print("No duplicated")
            blocks_unique = pd.concat([blocks_unique, block_id], ignore_index=True)
        else:
            print("Duplicated")
            blocks_value_duplicated = list(block_id["V"].duplicated(keep=False))
            blocks_value_duplicated_not = [not i for i in blocks_value_duplicated]
            all_different = all(blocks_value_duplicated_not)
            if all_different:
                print("All different")
                value_list = list(block_id["V"])
                value_random = random.choice(value_list)
                block_id_all = block_id.reset_index(drop=True)
                block_id_all = block_id_all.iloc[[0]]
                block_id_all["V"] = value_random
                blocks_unique = pd.concat([blocks_unique, block_id_all], ignore_index=True)
            else:
                print("Mode")
                block_id_mode = block_id.reset_index(drop=True)
                value_mode = stats.mode(block_id_mode["V"])[0][0]
                block_id_mode = block_id_mode.iloc[[0]]
                block_id_mode["V"] = value_mode
                blocks_unique = pd.concat([blocks_unique, block_id_mode], ignore_index=True)
    blocks_unique = blocks_unique[["X", "Y", "Z", "V"]]
    print("Creating file with block with value different than -999")
    Write_sgems(blocks_unique, filename_out_1[:-4], " ", filename_out_1)
    print("File Created: " + filename_out_1)
    blocks_unique_xmax = blocks_unique["X"].max()
    blocks_unique_ymax = blocks_unique["Y"].max()
    blocks_unique_zmax = blocks_unique["Z"].max()
    X = blocks_unique["X"].values
    Y = blocks_unique["Y"].values
    Z = blocks_unique["Z"].values
    V = blocks_unique["V"].values
    dictionary = {}
    print("Creating dictionary of blocks to simulate")
    for i in range(blocks_unique.shape[0]):
        dictionary[X[i], Y[i], Z[i]] = V[i]
    dictionary_len = len(dictionary)
    print("Active Blocks     :" + str(dictionary_len))
    b_n_x_a = np.linspace(1, blocks_unique_xmax, blocks_unique_xmax).astype(int)
    b_n_y_a = np.linspace(1, blocks_unique_ymax, blocks_unique_ymax).astype(int)
    b_n_z_a = np.linspace(1, blocks_unique_zmax, blocks_unique_zmax).astype(int)
    start = time.time()
    file = open(filename_out_2, "w+")
    file.write(
        filename_out_2[:-4]
        + "_"
        + str(blocks_unique_xmax)
        + "_"
        + str(blocks_unique_ymax)
        + "_"
        + str(blocks_unique_zmax)
        + "\n"
    )
    file.write("4" + "\n")
    file.write("x" + "\n")
    file.write("y" + "\n")
    file.write("z" + "\n")
    file.write("v" + "\n")
    for k in b_n_z_a:
        for j in b_n_y_a:
            for i in b_n_x_a:
                if (i, j, k) in dictionary:
                    file.write(str(i) + " " + str(j) + " " + str(k) + " " + repr(dictionary[i, j, k]) + "\n")
                else:
                    file.write(str(i) + " " + str(j) + " " + str(k) + " " + repr(-999) + "\n")
    file.close()
    end = time.time() - start
    print("Created file     :" + filename_out_2)
    print("It took (seconds):" + str(end))


def Blocks_to_distance(df_in, x_size, y_size, z_size):
    df_out = df_in.copy(deep=True)
    df_out["X"] = df_in["X"] * x_size
    df_out["Y"] = df_in["Y"] * y_size
    df_out["Z"] = df_in["Z"] * z_size
    return df_out


def Training_row(df_in, index, nnearest):
    block = df_in.iloc[[index]]
    block = block.reset_index(drop=True)
    temp_d = pd.DataFrame()
    temp_d = df_in[["X", "Y", "Z"]]
    temp_d["dX"] = df_in["X"] - block["X"][0]
    temp_d["dY"] = df_in["Y"] - block["Y"][0]
    temp_d["dZ"] = df_in["Z"] - block["Z"][0]
    temp_d["D"] = np.sqrt(temp_d["dX"] ** 2 + temp_d["dY"] ** 2 + temp_d["dZ"] ** 2)
    temp_d["V"] = df_in[["V"]]
    temp_d = temp_d.sort_values(["D"])
    temp_d = temp_d.reset_index(drop=True)
    temp_d = temp_d[1 : nnearest + 1]
    temp_d_row = temp_d.stack()
    temp_d_row.index = temp_d_row.index.map("{0[1]}_{0[0]}".format)
    temp_d_row = temp_d_row.to_frame().T
    temp_d_row["X"] = block["X"]
    temp_d_row["Y"] = block["Y"]
    temp_d_row["Z"] = block["Z"]
    temp_d_row["V"] = block["V"]
    return temp_d_row


def Features(df_in, block, nnearest):
    block = block.reset_index(drop=True)
    temp_d = pd.DataFrame()
    temp_d = df_in[["X", "Y", "Z"]]
    temp_d["dX"] = df_in["X"] - block["X"][0]
    temp_d["dY"] = df_in["Y"] - block["Y"][0]
    temp_d["dZ"] = df_in["Z"] - block["Z"][0]
    temp_d["D"] = np.sqrt(temp_d["dX"] ** 2 + temp_d["dY"] ** 2 + temp_d["dZ"] ** 2)
    temp_d["V"] = df_in[["V"]]
    temp_d = temp_d.sort_values(["D"])
    temp_d = temp_d.reset_index(drop=True)
    temp_d = temp_d[1 : nnearest + 1]
    temp_d_row = temp_d.stack()
    temp_d_row.index = temp_d_row.index.map("{0[1]}_{0[0]}".format)
    temp_d_row = temp_d_row.to_frame().T
    temp_d_row["X"] = block["X"]
    temp_d_row["Y"] = block["Y"]
    temp_d_row["Z"] = block["Z"]
    temp_d_row["V"] = block["V"]
    return temp_d_row


def Training_table(df_in, nnearest=10):
    table = pd.DataFrame()
    n = len(df_in)
    for i in range(n):
        perc = i / n * 100
        prec = 2
        print("Working on block: " + str(i) + " out of :" + str(n) + ", " + "{:.{}f}".format(perc, prec) + "%")
        row_i = Training_row(df_in, i, nnearest)
        table = pd.concat([table, row_i], ignore_index=True)
    return table


def Plot_trained(history, hyper_num, hyper_str, folder):
    acc_key = "acc" if "acc" in history.history else "accuracy"
    val_acc_key = "val_acc" if "val_acc" in history.history else "val_accuracy"
    plt.plot(history.history[acc_key])
    plt.plot(history.history[val_acc_key])
    title = hyper_str[4] + "\n"
    title = title + "Model Accuracy" + "\n"
    title = title + "categories: " + hyper_str[3] + "\n"
    title = title + "batch_size:" + hyper_num[0] + "\n"
    title = title + "epochs:" + hyper_num[1] + "\n"
    title = title + "points:" + hyper_num[2] + "\n"
    title = title + "activation:" + hyper_str[2] + "\n"
    title = title + "loss_function:" + hyper_str[0] + "\n"
    title = title + "optimizer:" + hyper_str[1] + "\n"
    title = title + "nodes:" + hyper_num[3] + "," + hyper_num[4] + "," + hyper_num[5] + "," + hyper_num[6]
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.ylim([0, 1])
    plt.tight_layout()
    filename_out = (
        hyper_str[3]
        + "_"
        + hyper_num[0]
        + "_"
        + hyper_num[1]
        + "_"
        + hyper_num[2]
        + "_"
        + hyper_str[2]
        + "_"
        + hyper_str[0]
        + "_"
        + hyper_str[1]
    )
    fn_out = "_"
    for i in hyper_num[3:]:
        fn_out = fn_out + i + "_"
    fn_out = filename_out + fn_out[:-1]
    plt.savefig(folder + fn_out + ".pdf")
    plt.close()
    return fn_out


def Plot_loss_trained_continuous(history, hyper_num, nodes_layer_str, hyper_str, folder):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    title = hyper_str[3] + "\n"
    title = title + "Model Loss" + "\n"
    title = title + "batch_size:" + hyper_num[0] + "\n"
    title = title + "epochs:" + hyper_num[1] + "\n"
    title = title + "points:" + hyper_num[2] + "\n"
    title = title + "activation:" + hyper_str[2] + "\n"
    title = title + "loss_function:" + hyper_str[0] + "\n"
    title = title + "optimizer:" + hyper_str[1] + "\n"
    nodes = "nodes:"
    for i in nodes_layer_str:
        nodes = nodes + i + ","
    title = title + nodes[:-1] + "\n"
    title = title + "training time [s] :" + hyper_num[3]
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.tight_layout()
    filename_out = (
        hyper_num[0]
        + "_"
        + hyper_num[1]
        + "_"
        + hyper_num[2]
        + "_"
        + hyper_str[2]
        + "_"
        + hyper_str[0]
        + "_"
        + hyper_str[1]
    )
    fn_out = "_"
    for i in nodes_layer_str:
        fn_out = fn_out + i + "_"
    fn_out = filename_out + fn_out[:-1]
    plt.savefig(folder + fn_out + "_loss.pdf")
    plt.close()
    return fn_out


def Plot_accu_trained_continuous(history, hyper_num, hyper_str, folder):
    acc_key = "acc" if "acc" in history.history else "accuracy"
    val_acc_key = "val_acc" if "val_acc" in history.history else "val_accuracy"
    plt.plot(history.history[acc_key])
    plt.plot(history.history[val_acc_key])
    title = hyper_str[3] + "\n"
    title = title + "Model Accuracy" + "\n"
    title = title + "batch_size:" + hyper_num[0] + "\n"
    title = title + "epochs:" + hyper_num[1] + "\n"
    title = title + "points:" + hyper_num[2] + "\n"
    title = title + "activation:" + hyper_str[2] + "\n"
    title = title + "loss_function:" + hyper_str[0] + "\n"
    title = title + "optimizer:" + hyper_str[1] + "\n"
    title = title + "nodes:" + hyper_num[3] + "," + hyper_num[4] + "," + hyper_num[5] + "," + hyper_num[6]
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.tight_layout()
    filename_out = (
        hyper_num[0]
        + "_"
        + hyper_num[1]
        + "_"
        + hyper_num[2]
        + "_"
        + hyper_str[2]
        + "_"
        + hyper_str[0]
        + "_"
        + hyper_str[1]
    )
    fn_out = "_"
    for i in hyper_num[3:]:
        fn_out = fn_out + i + "_"
    fn_out = filename_out + fn_out[:-1]
    plt.savefig(folder + fn_out + "_accu.pdf")
    plt.close()
    return fn_out


def Rotate_normal_z(df_in, angle):
    df_in["X2"] = df_in["X"] * np.cos(angle * np.pi / 180) - df_in["Y"] * np.sin(angle * np.pi / 180)
    df_in["Y2"] = df_in["X"] * np.sin(angle * np.pi / 180) + df_in["Y"] * np.cos(angle * np.pi / 180)
    df_in["X"] = df_in["X2"]
    df_in["Y"] = df_in["Y2"]
    df_in = df_in.drop(["X2", "Y2"], axis=1)
    return df_in


def Translate(df_in, x_min, y_min, z_min):
    df_in["X"] = df_in["X"] - x_min
    df_in["Y"] = df_in["Y"] - y_min
    df_in["Z"] = df_in["Z"] - z_min
    return df_in
