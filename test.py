import classifier
import pickle

def class_results():
    print "bbb"
    c1 = classifier.classifier()
    print "aaa"
    #c1.build_tf_idf()
    #c1.feature_selection()
    c1.build_tf_idf_feature_selection()
    c1.vectorization()
#    c1.calculate_df_in_classes()
#    c1.chi_square()
#    c1.chi_feature_list(50)

    svc1 = c1.svm_train_food()
    c1.svm_test_food(svc1)
    print "============the classification results for food are:=================== "
    
    for i in c1.food_results:
        print i['Title'],"\n", i['Content'],"\n"

    
    svc2 = c1.svm_train_movie()
    c1.svm_test_movie(svc2)
    print "=============the classification results for movie are: ==============="
    for i in c1.movie_results:
        print i['Title'],"\n", i['Content'],"\n"

    return c1.food_results, c1.movie_results
def main():
    (food, movie) = class_results()
    food_txt = pickle.load( open( "food.txt", "rb" ) )
    movie_txt = pickle.load( open( "movie.txt", "rb" ) )
    print "food:\n", food_txt
    print "movie: \n", movie_txt
    

if __name__ == '__main__':
    main()
