import pandas as pd
import numpy as np
from treelib import Node, Tree
df = pd.read_csv("agaricus-lepiota.data", header = None)

from sklearn.model_selection import train_test_split

df.columns = ['class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 
               'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 
               'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring', 
               'veil_type', 'veil_color', 'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']
df = df.drop(df[df.stalk_root  == "?"].index)
X = df.drop('class', axis=1)
y = df['class']


# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
training, testing = train_test_split(df, test_size=0.2, random_state=42)

# elements, counts = np.unique(df['odor'], return_counts = True)
# print(elements, counts)


# train_X.drop(['cap_color','odor'], axis= 1)
# train_X.columns[0]

class DecisionTree(Tree):
    """
    A class to represent a decision tree
    """
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Initialize a decision tree
        
        Parameters:
        max_depth (int): Maximum depth of the tree
        min_samples_split (int): Minimum number of samples required to split a node
        """
        self.depth = -1
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        super().__init__()

   
        

    def entropy(self, target_col):
        # elements, counts = np.unique(target_col, return_counts = True)
        # entropy = 0

        # for i in range(len(elements)):
        #    if counts[i] >0:
        #     p = counts[i]/ np.sum(counts)
        #     entropy += -p * np.log2(p)
        # return entropy
        elements,counts = np.unique(target_col,return_counts = True)
        entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return entropy
        
    def information_Gain(self, data,split_feature, target_col="class"):
        # total_entropy = self.entropy(data[target_col])

        # value, counts = np.unique(data[split_feature], return_counts = True)
        # weighted_entropy = 0
        # for i in range(len(value)):

        #     if counts[i] >0:
        #         p = counts[i]/ np.sum(counts)

        #         weighted_entropy += p * self.entropy(data.where(data[split_feature] == value[i]).dropna()[target_col])

        # information_gain = total_entropy - weighted_entropy
        # return information_gain
        total_entropy = self.entropy(data[target_col])

        #Calculate the values and the corresponding counts for the split attribute 
        vals,counts= np.unique(data[split_feature],return_counts=True)
        
        #Calculate the weighted entropy
        Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*self.entropy(data.where(data[split_feature]==vals[i]).dropna()[target_col]) for i in range(len(vals))])
        
        #Calculate the information gain
        Information_Gain = total_entropy - Weighted_Entropy
        return Information_Gain


    def build_tree(self, data,original_data,features, target_col ="class", parent_node_value = None,):
         
        if len(np.unique(data[target_col])) <= 1 or self.depth == self.max_depth:
            node_Value = np.unique(data[target_col])[0]
            return Node( node_Value )

        elif len(data) == 0:
            node_Value = np.unique(original_data[target_col])[np.argmax(np.unique(original_data[target_col],return_counts=True)[1])]
            return Node( node_Value)
        elif len(features) ==0:
            return Node(parent_node_value)

        else:

            parent_node_value = np.unique(data[target_col])[np.argmax(np.unique(data[target_col],return_counts=True)[1])]

            information_Gain_List =[]

            for feature in features:
                    information_Gain_List.append(self.information_Gain(data,feature))
            
            best_feature_index = np.argmax(information_Gain_List)
            best_feature = features[best_feature_index]


            #  = Tree()
            self.create_node(best_feature, best_feature)
            # self.create_node(node)
            sub_features = features.drop(best_feature)
            self.depth +=1
            print(best_feature)

            for value in np.unique(data[best_feature]):
                print(value)
                sub_data = data.where(data[best_feature] == value).dropna()

                sub_tree = self.build_tree(sub_data,original_data,sub_features,target_col,parent_node_value)

                self.paste(value, sub_tree)
            return tree

        # get best feature, create that node, then iterate through all the values of that feature and create a new node for each value. 
        # Then, for each value, create a new data set that has all the rows where the feature value is equal to the value of the node. 
        # Then, recursively call this function on the new data set and the new node.
        
tree = DecisionTree(max_depth= 3)
tree.build_tree(training, training, training.columns[1:])
tree.show()