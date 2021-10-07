#using pandas to read the data sets into pandas data frame
import pandas as pd
pd.options.mode.chained_assignment = None

#using matplot to diplay the graph
import matplotlib.pyplot as plt

# Reading the data set
pdf = pd.read_csv("/Users/rmkchowdary/Downloads/image_labels.csv")

#Data preparation & Cleaning
newdf = pdf[(pdf.label == 'Helmet') | (pdf.label == 'Helmet-Blurred')]


newdf['image'] = newdf['image'].str[ : 5]
# finding the number helmets in each vedio based on image frames 
finaldf = newdf.groupby("image").size()
finaldf = finaldf.reset_index(name = "helmet_count")

# Setting X and Y axis

finaldf.plot(x = "image" , y = "helmet_count" ,color = 'g')
plt.xlabel("Video")
plt.ylabel("helmet count")
# Plotting the Graph
plt.show()

