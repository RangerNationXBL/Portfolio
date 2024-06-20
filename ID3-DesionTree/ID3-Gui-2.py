import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import font as tkfont

#TODO: DocStrings and Notes

class ID3DecisionTree:
    def __init__(self, train_data, label):
        self.train_data = train_data
        self.label = label
        self.tree = None

    def calc_total_entropy(self):
        total_row = self.train_data.shape[0]
        total_entropy = 0
        class_list = self.train_data[self.label].unique()

        for c in class_list:
            total_class_count = self.train_data[self.train_data[self.label] == c].shape[
                0
            ]
            if total_class_count == 0:
                continue
            total_class_entropy = -(total_class_count / total_row) * np.log2(
                total_class_count / total_row
            )
            total_entropy += total_class_entropy

        return total_entropy

    def calc_entropy(self, feature_value_data):
        class_count = feature_value_data.shape[0]
        entropy = 0
        class_list = self.train_data[self.label].unique()

        for c in class_list:
            label_class_count = feature_value_data[
                feature_value_data[self.label] == c
            ].shape[0]
            if label_class_count == 0:
                continue
            probability_class = label_class_count / class_count
            entropy_class = -probability_class * np.log2(probability_class)
            entropy += entropy_class

        return entropy

    def calc_info_gain(self, feature_name):
        feature_value_list = self.train_data[feature_name].unique()
        total_row = self.train_data.shape[0]
        feature_info = 0.0

        for feature_value in feature_value_list:
            feature_value_data = self.train_data[
                self.train_data[feature_name] == feature_value
            ]
            feature_value_count = feature_value_data.shape[0]
            feature_value_entropy = self.calc_entropy(feature_value_data)
            feature_value_probability = feature_value_count / total_row
            feature_info += feature_value_probability * feature_value_entropy

        total_entropy = self.calc_total_entropy()
        info_gain = total_entropy - feature_info
        return info_gain

    def find_most_informative_feature(self):
        feature_list = self.train_data.columns.drop(self.label)
        max_info_gain = -1
        max_info_feature = None

        for feature in feature_list:
            feature_info_gain = self.calc_info_gain(feature)
            if max_info_gain < feature_info_gain:
                max_info_gain = feature_info_gain
                max_info_feature = feature

        return max_info_feature

    def generate_sub_tree(self, feature_name):
        feature_value_count_dict = self.train_data[feature_name].value_counts(
            sort=False
        )
        tree = {}
        class_list = self.train_data[self.label].unique()

        for feature_value, count in feature_value_count_dict.items():
            feature_value_data = self.train_data[
                self.train_data[feature_name] == feature_value
            ]
            assigned_to_node = False

            for c in class_list:
                class_count = feature_value_data[
                    feature_value_data[self.label] == c
                ].shape[0]
                if class_count == count:
                    tree[feature_value] = c
                    self.train_data = self.train_data[
                        self.train_data[feature_name] != feature_value
                    ]
                    assigned_to_node = True
                    break  # Move to next feature value once a class is assigned

            if not assigned_to_node:
                tree[feature_value] = "?"

        return tree

    def make_tree(self, root, prev_feature_value):
        if self.train_data.shape[0] == 0:
            return

        max_info_feature = self.find_most_informative_feature()
        if max_info_feature is None:
            return

        tree = self.generate_sub_tree(max_info_feature)
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
                feature_value_data = self.train_data[
                    self.train_data[max_info_feature] == node
                ]
                if feature_value_data.empty:
                    next_root[node] = "Unknown"
                else:
                    self.train_data = feature_value_data
                    self.make_tree(next_root, node)

    def fit(self):
        self.tree = {}
        self.make_tree(self.tree, None)
        return self.tree

    def predict(self, instance):
        current_tree = self.tree
        while isinstance(current_tree, dict):
            root_node = next(iter(current_tree))
            if root_node in instance:
                feature_value = instance[root_node]
                if feature_value in current_tree[root_node]:
                    current_tree = current_tree[root_node][feature_value]
                else:
                    return None
            else:
                return None
        return current_tree

    def evaluate(self, test_data):
        correct_predict = 0
        wrong_predict = 0

        for index, row in test_data.iterrows():
            result = self.predict(test_data.iloc[index])
            if result == test_data[self.label].iloc[index]:
                correct_predict += 1
            else:
                wrong_predict += 1

        accuracy = correct_predict / (correct_predict + wrong_predict)
        return accuracy


class ID3DecisionTreeVisualizer(ID3DecisionTree):
    def __init__(self, train_data, label):
        super().__init__(train_data, label)
        self.steps = "Steps to build the decision tree:\n"

    def fit(self):
        tree = super().fit()
        self.steps += f"Generated Decision Tree:\n{tree}\n\n"
        return tree

    def visualize_tree(self, tree, steps, accuracy):
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

        self.draw_tree(tree, canvas)

        root.mainloop()

    def draw_tree(
        self, tree, canvas, x=400, y=50, dx=200, dy=50, parent=None, parent_text=None
    ):
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
            canvas.create_text((parent[0] + x) / 2, (parent[1] + y) / 2, text=parent_text, fill="black", font=custom_font,)

        for i, (key, value) in enumerate(children.items()):
            self.draw_tree(value, canvas, x - dx + i * (dx // len(children)), y + dy, dx // 2, dy, (x, y), key,)


# Read the training and test data
train_data = pd.read_csv("D:/JustProgramming/ID3-DesionTree/PlayTennis.csv")
test_data = pd.read_csv("D:/JustProgramming/ID3-DesionTree/PlayTennis.csv")

# Generate the decision tree
visualizer = ID3DecisionTreeVisualizer(train_data, "Play Tennis")
tree = visualizer.fit()

# Evaluate the decision tree
accuracy = visualizer.evaluate(test_data)
visualizer.steps += f"Accuracy: {accuracy:.2f}\n"

# Visualize the decision tree with accuracy and steps
visualizer.visualize_tree(tree, visualizer.steps, accuracy)
