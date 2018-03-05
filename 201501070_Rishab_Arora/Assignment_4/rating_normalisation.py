import csv

out=open("ratings_woheader.csv","r")
data=csv.reader(out)
data=[[row[0],row[1],eval(row[2])] for row in data]
out.close()

#print(data)

normalised_data=[[row[0], row[1], row[2]/5] for row in data]
#print(normalised_data)

out=open("small_normalised_ratings.csv","w")
output=csv.writer(out)

for row in normalised_data:
	if eval(row[0])<=1000 and eval(row[1])<=5000:
		output.writerow(row)
	
out.close()
