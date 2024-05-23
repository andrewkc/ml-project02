import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.decomposition import PCA
from pyts.transformation import ROCKET
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import classification_report

'''
El uso de árboles de decisión para analizar datos EEG en el contexto de la predisposición al alcoholismo 
puede ser recomendado debido a su interpretabilidad, capacidad para manejar relaciones no lineales, selección 
automática de características, manejo de datos faltantes y valores atípicos, eficiencia con grandes 
volúmenes de datos, y su capacidad para ser mejorado con técnicas de ensamble. Estos atributos hacen que 
los árboles de decisión sean una opción adecuada y poderosa para este tipo de análisis complejo.
Sin embargo, se obtine un accuracy de 0.73, menor al modelo de Regresiòn Logìstica y el SVM, esto puede 
suceder a la sensibilidad de las características, el desbalanceo en las clases y a que los parámetros
no estén bien ajustados.
'''
class Nodo:
 #Define what your data members will be
  def __init__(self, X, Y, index=None, threshold=None):
    self.X = X
    self.Y = Y
    self.index = index
    self.threshold = threshold
    self.left = None
    self.right = None

  def IsTerminal(self):
    # return true if this node has the same labels in Y
    return np.all(self.Y == self.Y[0])


  def BestSplit(self):
    # Determine the best split for the node data based on Gini impurity
    best_index = None
    best_value = None
    best_score = float("inf")

    for index in range(self.X.shape[1]):  # iterate over each feature
      values = self.X[:, index]
      for value in np.unique(values):
        left_mask = values <= value
        right_mask = values > value
        left_y = self.Y[left_mask]
        right_y = self.Y[right_mask]

        # Calculate weighted Gini for each split
        left_gini = self.Gini(left_y)
        right_gini = self.Gini(right_y)
        weighted_gini = (len(left_y) / len(self.Y)) * left_gini + (len(right_y) / len(self.Y)) * right_gini

        if weighted_gini < best_score:
          best_score = weighted_gini
          best_index = index
          best_value = value

    return best_index, best_value, best_score

  def Entropy(self):
    _, counts = np.unique(self.Y, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = - np.sum(probabilities * np.log2(probabilities))
    return entropy

  def Gini(self, Y):
    _ , counts = np.unique(Y, return_counts=True)
    #print(counts)
    probabilities = counts / counts.sum()
    #print(probabilities)
    gini = 1 - np.sum(probabilities**2)
    return gini

class DT:
    def __init__(self, X, Y):
        self.m_Root = None
        self.X = X
        self.Y = Y

    def create_DT(self):
        self.m_Root = self._create_tree(self.X, self.Y)

    def _create_tree(self, X, Y):
        node = Nodo(X, Y)
        if node.IsTerminal():
            return node

        # Utiliza BestSplit que a su vez utiliza Gini dentro de la clase Nodo
        index, value, _ = node.BestSplit()
        if index is None:
            return node

        # Crear nodos hijos basados en la mejor división
        left_mask = X[:, index] <= value
        right_mask = X[:, index] > value
        left_child = self._create_tree(X[left_mask], Y[left_mask])
        right_child = self._create_tree(X[right_mask], Y[right_mask])

        # Configurar el nodo actual con los detalles de la división y los nodos hijos
        node.index = index
        node.threshold = value
        node.left = left_child
        node.right = right_child
        return node

    def predict(self, x):
        # Predice la clase de una observación utilizando el árbol creado
        return self._predict_node(self.m_Root, x)

    def _predict_node(self, node, x):
        if node.left is None and node.right is None:
            # Nodo terminal, retorna la clase más frecuente
            node_y_int = node.Y.astype(np.int64, casting='unsafe')
            return np.bincount(node_y_int).argmax()
        elif x[node.index] <= node.threshold:
            return self._predict_node(node.left, x)
        else:
            return self._predict_node(node.right, x)


def confusionMatrix(y_pred, y_test, type_label):
  labels = ['no predisposition to alcoholism', 'predisposition to alcoholism']
  matrix = confusion_matrix(y_test, y_pred).astype('float')
  matrix = matrix / matrix.sum(axis=1)[:, np.newaxis]
  df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)

  sns.heatmap(df_matrix, annot=True, cbar=False, cmap="Greens")
  plt.title("Confusion Matrix " + type_label)
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.tight_layout()
  plt.show()

def getMetrics(y_test, y_pred):
  targets = ['no predisposition to alcoholism', 'predisposition to alcoholism']

  precision = precision_score(y_test, y_pred, average=None)
  f1 = f1_score(y_test, y_pred, average=None)

  report = classification_report(y_test, y_pred, target_names=targets)
  print("My Model Metrics:")
  print(report)

  results = {}
  for i, name in enumerate(targets):
    results[name] = {
      "Precision": precision[i],
      "F1-Score": f1[i]
    }

  return results

def readH5File(filename):
  with h5py.File(filename, 'r') as f:
    x = f['x'][:]
    if 'y' in f:
      y = f['y'][:]
      return x, y
    return x

X_train, Y_train = readH5File('train.h5')
X_test = readH5File('test.h5')

# Reduce dimensionality X_train and X_test
X_train = X_train.squeeze(axis=1)
X_test = X_test.squeeze(axis=1)

# Split
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)

# Normalize x_train and x_test before applying ROCKET
scaler = StandardScaler()
x_train_normalized = scaler.fit_transform(x_train)
x_test_normalized = scaler.fit_transform(x_test)

# Applying ROCKET
rocket = ROCKET()
rocket.fit(x_train_normalized)
x_train_transformed = rocket.transform(x_train_normalized)
x_test_transformed = rocket.transform(x_test_normalized)

y_train_transformed = (y_train + 1) // 2

dt = DT(x_train_transformed, y_train_transformed)
dt.create_DT()


predictions = np.array([dt.predict(x) for x in x_test_transformed])
y_test = (y_test + 1) // 2

print(len(y_test))

#accuracy
print(sum(predictions == y_test)/len(y_test))

print(predictions, y_test)

confusionMatrix(predictions, y_test, type_label='DT')

results = getMetrics(y_test, predictions)