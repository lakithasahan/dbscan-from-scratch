from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_iris
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances,f1_score,precision_score,recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

#Custom estimator for gridsearch 
class MyClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,e=0,minp=0):
       
        self.e =e
        self.minp=minp
        
        
    def fit(self, X,Y):
		self.Y=Y
		#print(self.Y)
		DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, 'euclidean'))
		#print(DistanceMatrix)

		core_point_array=np.zeros(150)
		cluster_array=np.zeros(150)
		PointNeighbors=[]
		
		e=self.e
		k=self.minp
		#print(e)
		w=0
		for i in range(len(DistanceMatrix)):
	
			PointNeighbors=np.where(DistanceMatrix[i]<=e)[0]
			if len(PointNeighbors)>=k:
				core_point_array[i]=1
				if cluster_array[i]==0:
					cluster_array[i]=w
					w=w+1
		
				for x in range(len(PointNeighbors)):
					#print(cluster_array[PointNeighbors[x]])	
					if cluster_array[PointNeighbors[x]]==0:
						cluster_array[PointNeighbors[x]]=cluster_array[i]
		
		
		
			
				#print(PointNeighbors)
		
	
		for x in range(len(cluster_array)):
				cluster_array[x]=cluster_array[x]-1	


	
		#print('Number of core points -'+str( np.count_nonzero(core_point_array)))	
		#print('Number of clusters -'+str( np.count_nonzero(cluster_array)))	

		#print(target_data)
		#print(core_point_array)
		#print(cluster_array)
		
		self.cluster_array=cluster_array
		return cluster_array
       
       
       
    def predict(self, X):
         # Some code
         return self.cluster_array 


    def score(self, X, Y):
	
		dt=f1_score(self.Y,self.cluster_array,average='weighted')
		print('Accuracy -'+str(dt))
		return (dt)

        
        
def DBSCAN(normalised_distance,e,k):


	DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(normalised_distance, 'euclidean'))
	#print(DistanceMatrix)

	core_point_array=np.zeros(150)
	cluster_array=np.zeros(150)
	PointNeighbors=[]
	#e=0.3
	#k=18
	w=0
	for i in range(len(DistanceMatrix)):
	
		PointNeighbors=np.where(DistanceMatrix[i]<=e)[0]
		if len(PointNeighbors)>=k:
			core_point_array[i]=1
			if cluster_array[i]==0:
				cluster_array[i]=w
				w=w+1
		
			for x in range(len(PointNeighbors)):
				#print(cluster_array[PointNeighbors[x]])	
				if cluster_array[PointNeighbors[x]]==0:
					cluster_array[PointNeighbors[x]]=cluster_array[i]
		
		
		
			
			#print(PointNeighbors)
		
	
	for x in range(len(cluster_array)):
			cluster_array[x]=cluster_array[x]-1	


	
	#print('Number of core points -'+str( np.count_nonzero(core_point_array)))	
	#print('Number of clusters -'+str( np.count_nonzero(cluster_array)))	

	print(target_data)
	#print(core_point_array)
	print(cluster_array)

	return cluster_array


###################################################################################################
#Getting iris data

iris =load_iris()
input_data=iris.data
target_data=iris.target

###################################################################################################
#Data Manipulations before introducing to the algorithm

poly = PolynomialFeatures(2)
input_data=poly.fit_transform(input_data)  
#print(input_data)

input_data=QuantileTransformer(n_quantiles=40, random_state=0).fit_transform(input_data)

scaler = MinMaxScaler()
 
scaler.fit(input_data)
normalised_input_data=scaler.transform(input_data)

distan=pairwise_distances(normalised_input_data,metric='euclidean')


scaler.fit(distan)
normalised_distance=scaler.transform(distan)


sscaler = StandardScaler()
sscaler.fit(normalised_distance)
normalised_distance=sscaler.transform(normalised_distance)


pca = PCA(n_components=4)
normalised_distance = pca.fit_transform(normalised_distance)

scaler.fit(normalised_distance)
normalised_distance=scaler.transform(normalised_distance)


print(normalised_distance)
print('normalised_distance')


##############################################################################################
#Training the algorithm using GridSearch



eps_values= np.arange(0.1,0.5 ,0.001)
min_sample_values = np.arange(2,30,1)

params = {
    'e':eps_values,
    'minp':min_sample_values
}
cv = [(slice(None), slice(None))]
gs = GridSearchCV(MyClassifier(), param_grid=params, cv=cv)

Y=target_data
gs.fit(normalised_distance,Y)

print(gs.best_params_)

para=gs.best_params_



#############################################################################################
#Testing the best selected parameters by plotting 


e=para['e']
k=para['minp']


cluster_array=DBSCAN(normalised_distance,e,k)


print(target_data)
print(cluster_array.astype(int))

print('precision_score- '+str(precision_score(target_data,cluster_array,average='weighted',labels=np.unique(cluster_array))))
print('recall_score- '+str(recall_score(target_data,cluster_array,average='weighted',labels=np.unique(cluster_array))))




plt.subplot(2, 1, 1)
plt.scatter(normalised_distance[:,0], normalised_distance[:,1],c=cluster_array, cmap='Paired')
plt.title("custom DBSCAN predicted cluster outputs")

plt.subplot(2, 1, 2)
plt.scatter(normalised_distance[:,0], normalised_distance[:,1],c=target_data, cmap='Paired')
plt.title("Actual target outputs")

plt.tight_layout()
plt.show()




















