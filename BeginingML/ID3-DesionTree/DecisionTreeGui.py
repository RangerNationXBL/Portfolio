import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import font as tkfont

# Read the training data
train_data = pd.read_csv("D:/JustProgramming/ID3-DesionTree/PlayTennis.csv")


# Functions to generate and evaluate the ID3 decision tree
def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0]
    total_entropy = 0

    for c in class_list:
        total_class_count = train_data[train_data[label] == c].shape[0]
        if total_class_count == 0:
            continue
        total_class_entropy = -(total_class_count / total_row) * np.log2(
            total_class_count / total_row
        )
        total_entropy += total_class_entropy

    return total_entropy


def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0

    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]
        if label_class_count == 0:
            continue
        probability_class = label_class_count / class_count
        entropy_class = -probability_class * np.log2(probability_class)
        entropy += entropy_class

    return entropy


def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0

    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list)
        feature_value_probability = feature_value_count / total_row
        feature_info += feature_value_probability * feature_value_entropy

    total_entropy = calc_total_entropy(train_data, label, class_list)
    info_gain = total_entropy - feature_info
    return info_gain


def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label)
    max_info_gain = -1
    max_info_feature = None

    for feature in feature_list:
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature

    return max_info_feature


def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False)
    tree = {}

    for feature_value, count in feature_value_count_dict.items():
        feature_value_data = train_data[train_data[feature_name] == feature_value]

        assigned_to_node = False
        for c in class_list:
            class_count = feature_value_data[feature_value_data[label] == c].shape[0]

            if class_count == count:
                tree[feature_value] = c
                train_data = train_data[train_data[feature_name] != feature_value]
                assigned_to_node = True
                break  # Move to next feature value once a class is assigned

        if not assigned_to_node:
            tree[feature_value] = "?"

    return tree, train_data


def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0:
        max_info_feature = find_most_informative_feature(train_data, label, class_list)
        tree, train_data = generate_sub_tree(
            max_info_feature, train_data, label, class_list
        )
        next_root = None

        if prev_feature_value is not None:
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else:
            root[max_info_feature] = tree
            next_root = root[max_info_feature]

        for node, branch in list(next_root.items()):
            if branch == "?":
                feature_value_data = train_data[train_data[max_info_feature] == node]
                make_tree(next_root, node, feature_value_data, label, class_list)


def id3(train_data_m, label):
    train_data = train_data_m.copy()
    tree = {}
    class_list = train_data[label].unique()
    make_tree(tree, None, train_data, label, class_list)
    return tree


def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    else:
        root_node = next(iter(tree))
        feature_value = instance[root_node]
        if feature_value in tree[root_node]:
            return predict(tree[root_node][feature_value], instance)
        else:
            return None


def evaluate(tree, test_data_m, label):
    correct_predict = 0
    wrong_predict = 0

    for index, row in test_data_m.iterrows():
        result = predict(tree, test_data_m.iloc[index])
        if result == test_data_m[label].iloc[index]:
            correct_predict += 1
        else:
            wrong_predict += 1

    accuracy = correct_predict / (correct_predict + wrong_predict)
    return accuracy


# Visualize the decision tree using tkinter
def draw_tree(tree, canvas, x=400, y=50, dx=200, dy=50, parent=None, parent_text=None):
    if not isinstance(tree, dict):
        canvas.create_text(x, y, text=tree, fill="black", font=custom_font)
        if parent:
            canvas.create_line(parent[0], parent[1], x, y, arrow=tk.LAST)
            canvas.create_text(
                (parent[0] + x) / 2,
                (parent[1] + y) / 2,
                text=parent_text,
                fill="black",
                font=custom_font,
            )
        return

    root = next(iter(tree))
    children = tree[root]
    canvas.create_text(x, y, text=root, fill="black", font=custom_font)
    if parent:
        canvas.create_line(parent[0], parent[1], x, y, arrow=tk.LAST)
        canvas.create_text(
            (parent[0] + x) / 2,
            (parent[1] + y) / 2,
            text=parent_text,
            fill="black",
            font=custom_font,
        )

    for i, (key, value) in enumerate(children.items()):
        draw_tree(
            value,
            canvas,
            x - dx + i * (dx // len(children)),
            y + dy,
            dx // 2,
            dy,
            (x, y),
            key,
        )


def visualize_tree(tree, steps, accuracy):
    root = tk.Tk()
    root.title("Decision Tree")

    global custom_font
    custom_font = tkfont.Font(family="Helvetica", size=12)

    # Create and pack accuracy label
    accuracy_label = tk.Label(
        root, text=f"Accuracy: {accuracy:.2f}", font=("Helvetica", 16, "bold")
    )
    accuracy_label.pack()

    # Create and pack canvas for the tree
    canvas = tk.Canvas(root, width=800, height=600, bg="#f0f0f0")
    canvas.pack()

    # Create and pack text widget for steps
    steps_text = tk.Text(root, height=10, wrap="word", font=custom_font)
    steps_text.pack()
    steps_text.insert(tk.END, steps)
    steps_text.config(state=tk.DISABLED)  # Make the text widget read-only

    draw_tree(tree, canvas)

    root.mainloop()


# Generate the decision tree
train_data_m = pd.read_csv("D:/JustProgramming/ID3-DesionTree/PlayTennis.csv")
steps = "Steps to build the decision tree:\n"
tree = id3(train_data_m, "Play Tennis")

# Evaluate the decision tree
test_data_m = pd.read_csv("D:/JustProgramming/ID3-DesionTree/PlayTennis.csv")
accuracy = evaluate(tree, test_data_m, "Play Tennis")

# Append steps to the steps variable for displaying
steps += f"Generated Decision Tree:\n{tree}\n\n"
steps += f"Accuracy: {accuracy:.2f}\n"

print("Generated Decision Tree:\n", tree, "\n")
print(f"Accuracy: {accuracy}")

# Visualize the decision tree with accuracy and steps
visualize_tree(tree, steps, accuracy)
