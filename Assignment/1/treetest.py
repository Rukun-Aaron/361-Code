import pandas as pd
import numpy as np
from treelib import Node, Tree
from pprint import pprint
df = pd.read_csv("agaricus-lepiota.data", header = None)


from sklearn.model_selection import train_test_split

df.columns = ['class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 
               'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 
               'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring', 
               'veil_type', 'veil_color', 'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']
df = df.drop(df[df.stalk_root  == "?"].index)
X = df.drop('class', axis=1)
y = df['class']

training, testing = train_test_split(df, test_size=0.2, random_state=42)

# class Node():
#     def __init__(self, value):
#       self.value = value
#       self.depth = 0
#       self.children = []
#       self.parent=None

#     def add_child(self, item):
#       self.children.append(item)
#       item.depth=self.depth+1
#       item.parent=self
      


def entropy(target_col):
        elements,counts = np.unique(target_col,return_counts = True)
        entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return entropy

def information_Gain(data,split_feature, target_col="class"):
        total_entropy = entropy(data[target_col])
        vals,counts= np.unique(data[split_feature],return_counts=True)
        weighted_entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_feature]==vals[i]).dropna()[target_col]) for i in range(len(vals))])
        information_gain = total_entropy - weighted_entropy
        return information_gain

def build_tree(data,features, target_col="class",parent_node_value = None,depth=-1, max_depth=4, tree =None):
        
        if len(np.unique(data[target_col])) <= 1 or depth == max_depth :
            
            value = np.unique(data[target_col])[0]
            tree.create_node(value , identifier=len(tree.all_nodes())+1, parent=parent_node_value)
            return tree
           
        elif len(data) == 0:
          
            
            # node_Value = np.unique(original_data[target_col])[np.argmax(np.unique(original_data[target_col],return_counts=True)[1])]
            node_Value = np.unique(data[target_col])[np.argmax(np.unique(data[target_col],return_counts=True)[1])]

            tree.create_node(node_Value, node_Value, parent=parent_node_value)
            return tree
        
        elif len(features) ==0:
            
            return tree 
            
        else:
            
            # parent_node_class = np.unique(data[target_col])[np.argmax(np.unique(data[target_col],return_counts=True)[1])]

            information_Gain_List =[]

            for feature in features:
                    information_Gain_List.append(information_Gain(data,feature))
            
            best_feature_index = np.argmax(information_Gain_List)
            best_feature = features[best_feature_index]
            # item_values = [information_Gain(data,feature, target_col) for feature in features] #Return the information gain values for the features in the dataset
            # best_feature_index = np.argmax(item_values)
            # best_feature = features[best_feature_index]
          
            if tree is None:
                decision_tree = Tree()
                decision_tree.create_node(tag=best_feature, identifier=best_feature, parent=None)
            else:
                decision_tree = tree
                decision_tree.create_node(tag=best_feature, identifier=best_feature , parent=parent_node_value)
            
            features = features.drop(best_feature)
            # features = [i for i in features if i != best_feature]
        
        #Grow a branch under the root node for each possible value of the root node feature
            depth += 1
            print(np.unique(data[best_feature]))
            for value in np.unique(data[best_feature]):
                
                #Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
                sub_data = data.where(data[best_feature] == value).dropna()
                identifier = str(best_feature) + "_" + str(value)
                decision_tree.create_node(tag=value, identifier=identifier, parent=best_feature)
                build_tree(sub_data,features,target_col,identifier, depth, max_depth, decision_tree)
                
          
            return(decision_tree) 
def predict(df, tree):  
    
    node = tree.get_node(tree.root)
    while not node.is_leaf():
        
        feature = node.tag 
        value = df[feature]
        
        tree_feature = tree.get_node(feature + "_" + value)
        children = tree.children(tree_feature.identifier)
        if (children[0].is_leaf()):
             return children[0].tag
        else:
             
            node = children[0]
         

def test(df, tree):
    correct = 0
    for i in range(len(df)):
        predicted = predict(df.iloc[i], tree)
        ground_truth = df.iloc[i]["class"]
        if predicted == ground_truth:
            correct += 1
            # print(correct)
        else:
            print("wrong")
    return np.divide(correct,len(df))
x = build_tree(training, training.columns[1:])
x.show()
print(testing.info())
print(test (testing, x))

# print(x.get_node("odor_n"))
# print(x.children("odor_l")[0].tag)