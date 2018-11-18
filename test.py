from pyspark import SparkContext 
import csv
from pyspark.mllib.recommendation import ALS
import math

sc = SparkContext("local")
reviews_raw = sc.textFile("file:////home/vijayccbd/spark-2.3.1-bin-hadoop2.7/Reviews.csv")
reviews_raw_header = reviews_raw.take(1)[0]
reviews_data = reviews_raw.filter(lambda line: line!=reviews_raw_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (str(tokens[1]),str(tokens[0]),str(tokens[2]))).cache()
productmap_raw = sc.textFile("file:////home/vijayccbd/spark-2.3.1-bin-hadoop2.7/productmap.csv")
productmap_headers = productmap_raw.take(1)[0]
productmap_data = productmap_raw.filter(lambda line : line != productmap_headers).map(lambda line : line.split(",")).map(lambda tokens : (str(tokens[0]),tokens[1]))

print "OUTPUT:reviews data is {}".format(reviews_data.take(3))

training_RDD, validation_RDD, test_RDD = reviews_data.randomSplit([6, 2, 2], seed=0L)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

seed = 5L
iterations = 10
regularization_parameter = 0.1
errors = [0]*50
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in range(1,51):
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print 'OUTPUT:For rank {} the RMSE is {}'.format(rank, error)
    if error < min_error:
        min_error = error
        best_rank = rank

print 'OUTPUT:The best model was trained with rank {}'.format(rank)

# write the errors to a csv file
fp = open("errors.csv","w+")
writer = csv.writer(fp,delimiter = ",")
for i in errors:
    writer.writerow([i])

model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print 'OUTPUT:For testing data the RMSE is {}'.format(error)

# now we need to make recommend products

user_id = 181514

# we need to get all the products the user did not reviews

user_id_rows = reviews_data.filter(lambda x : int(x[0]) == user_id).map(lambda x : x[1]).collect() # make it a list

print "OUTPUT: user_id_rows are {}".format(user_id_rows[:10])

# now get all the products not rated by this user

user_id_predict_rows = (productmap_data.filter(lambda x : x[0] not in user_id_rows).map(lambda x : (str(user_id),str(x[0]))))

print "OUTPUT: user_id_predict_rows is {}".format(user_id_predict_rows.take(5))

user_recommendations = model.predictAll(user_id_predict_rows).map(lambda r : (r.product,r.rating)).takeOrdered(5,key = lambda x : -x[1])

top3 = user_recommendations[:3]

print "OUTPUT: user recommendations are {}".format(top3)

prdct_list = []
productmap_list = productmap_data.collect()

for prdct in top3:
    prdct_no = prdct[0]
    for p in productmap_list:
	if prdct_no == int(p[0]):
	    prdct_list.append(p[1])

print "OUTPUT: the recommended product list is {}".format(prdct_list)


