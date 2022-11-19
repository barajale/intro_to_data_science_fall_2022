### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ c4c85f7a-4c12-11ed-266d-a71f1fde33f8
using CSV, DataFrames, ScikitLearn, MolecularGraphKernels, CairoMakie, ColorSchemes, GraphViz, RDKitMinimalLib, PlutoUI, FileIO, Random

# ╔═╡ 3b6d11a6-4bb2-492a-8bd4-24e9c7bf719b
md"Alexander Barajas-Ritchie"

# ╔═╡ 51633144-4549-438b-9173-29918bcda346
md"# classifying a molecule as smelling fruity or not via a decision tree

!!! note \"our objective\"
	our objective is to train and test a decision tree classifier that predicts if a molecule smells fruity, or not. to enable a data-driven approach to predict the olfactory perception of a molecule, we use the combined Goodscents and Leffingwell data sets, where experts labeled molecules with olfactory descriptors. 

"

# ╔═╡ 04286557-a3e4-4858-bbc4-c99b1e88821a
TableOfContents()

# ╔═╡ 89020e90-caaa-4758-91f8-ff775af36c1b
set_theme!(theme_minimal()); update_theme!(fontsize=20)

# ╔═╡ b99239b9-4d53-4c1e-a8a3-f6374f8e6795
import MLJBase: partition, StratifiedCV, train_test_pairs

# ╔═╡ 9631ebda-9ca5-4d98-b956-feba97b6bef4
md"!!! note
	we will call the machine learning library in Python, `scikit-learn`, from Julia, since it is such an excellent and widely-used package for machine learning.
	* consult the `scikit-learn` Python docs [here](https://scikit-learn.org/stable/) for information on the functions, classes, and methods (note, Python is much more object-oriented than Julia...).
	* consult the `Scikitlearn.jl` Julia docs [here](https://cstjean.github.io/ScikitLearn.jl/dev/man/python/) to see how to import `scikit-learn` functions into Julia. below, I do this for you.
"

# ╔═╡ c64b517f-1351-48c5-8fab-73ab5294334e
 begin
	# import functions from scikitlearn
	@sk_import metrics : confusion_matrix
	@sk_import tree : DecisionTreeClassifier
	@sk_import tree : plot_tree
end

# ╔═╡ 17f913e8-0b76-463f-8354-02744f096ab6
md"## read in raw data

the data table `fruity.csv` [here](https://raw.githubusercontent.com/SimonEnsemble/intro_to_data_science_fall_2022/main/hw/dt_fruity_smell/fruity.csv) lists a set of molecules (well, their SMILES representations) and whether or not they smelled fruity to an olfactory expert. this labeled data is from combining the Goodscents and Leffingwell data sets as deposited in the [pyrfume project](https://pyrfume.org/). 

🥝 read the data into Julia as a `DataFrame`.
"

# ╔═╡ eab25b4b-3ab9-4b65-89b8-f3eaf535282a
begin
	download("https://raw.githubusercontent.com/SimonEnsemble/intro_to_data_science_fall_2022/main/hw/dt_fruity_smell/fruity.csv", "data.csv")
	data = CSV.read("data.csv", DataFrame)
	odor = "fruity"
end

# ╔═╡ 0ae7bc0b-689e-4b34-8783-bb518bf3d7de
data

# ╔═╡ 793170c3-9031-415a-b592-b86735f3272e
md"🥝 how many uniuque molecules are in the data set?"

# ╔═╡ 070ff73c-2159-419e-aeaf-76118eb92f73
length(unique(data[:,"molecule"]))

# ╔═╡ 700584de-1b6c-4414-ac90-2137e5566095
md"🥝 use `combine` and `groupby` to determine how many molecules have the fruity olfactory label and how many do not."

# ╔═╡ 71d32941-d7f6-45bd-b347-5717c65d3b29
combine(groupby(data, "fruity odor"), "fruity odor" => length => "fruity count")

# ╔═╡ fcf7c6d5-dd8c-4d9f-8c5a-c4a13f33a44a
md"## featurizing the molecules to obtain `X`

a decision tree takes as input a fixed-size vector representation of each example (here, a molecule). 

!!! note
	to learn more about MACCS fingerprints, see [here](https://chem.libretexts.org/Courses/Intercollegiate_Courses/Cheminformatics_OLCC_(2019)/06%3A_Molecular_Similarity/6.01%3A_Molecular_Descriptors).

🥝 convert each molecule into a vector by computing its MACCS fingerprint, a length-166 bit vector indicating the presence/absence of a list of predefined substructure patterns. store the feature vectors of the molecules in the rows of a (# molecules × 166) matrix, `X`. use the function from `MolecularGraphKernels.jl` [here](https://github.com/SimonEnsemble/MolecularGraphKernels.jl/blob/main/src/maccs.jl#L199-L211).

!!! warning
	unfortunately, RDKit is not built for Windows. Cory will provide this matrix to you.
"

# ╔═╡ c2528ff3-de8b-4f3c-979a-88c5f3e21d1b
begin
	X = falses(nrow(data), 166)
	data[!,"maccs_fp_count"] = zeros(length(data[:,"molecule"]))
	
	for (i, row) in enumerate(eachrow(data))
		X[i, :] = MolecularGraphKernels.maccs_fp(row["molecule"])
		data[i,"maccs_fp_count"] = sum(X[i,:])
	end
end

# ╔═╡ 2e906f65-6175-4681-9441-190b1f536ba0
X

# ╔═╡ 9d0aae10-caa5-4d44-a095-fadf5bb97d37
md"🥝 use the `heatmap` function to visualize the MACCS fingerprints of the molecules. this will show you which bits are on and off."

# ╔═╡ a53cbbf9-0b4e-4cc1-a174-4086ecb85b63
heatmap(transpose(X))

# ╔═╡ 1ae86cb8-6da7-47ba-8c03-42bc485396b8
md"🥝 visualize the distribution of the # of MACCS bits activated per molecule."

# ╔═╡ b3b795ff-7439-4b74-9d67-eef24c5b49a4
begin
	local fig = Figure()
	local ax = Axis(fig[1, 1])
	hist!(data[:, "maccs_fp_count"], color="limegreen", strokecolor=:black, strokewidth=1)
	ylims!(0, nothing)
	fig
end

# ╔═╡ 25200c84-03a2-4f71-b480-ec206976abbf
md"🥝 visualize the distribution of the # of molecules in the data set that activate a MACCS bit."

# ╔═╡ a1246cc6-da21-4576-859c-d88c87b45cc9
begin
	count_vector = zeros(166)
	for (i, col) in enumerate(eachcol(X))
		count_vector[i] = sum(X[:,i])
	end
	local fig = Figure()
	local ax = Axis(fig[1, 1],
		title="distribution of the # of molecules in the data set that activate a MACCS bit", 
		xlabel="bits",
	    ylabel="Number of Molecules",
		xticks = collect(0:5:166),
		xticklabelsize = 10

		
	)
	barplot!(1:length(count_vector), count_vector, color="blue",
		strokecolor=:black, strokewidth=1)
	ylims!(0, nothing)
	fig
end

# ╔═╡ 49ffc81a-f1c2-4b70-aa2d-ce67f4f62808
X

# ╔═╡ df61f5aa-ff37-46c5-8c0b-a5e4ddb9ae6c
md"## the target vector, `y`

🥝 construct the target vector `y` whose element `i` gives `true` if molecule `i` smelled fruity and `false` if it didn't. note, element `i` of `y` must correspond with row `i` of your feature matrix `X`.

!!! hint
	don't overthink this one---just pull a column out of the data frame!
"

# ╔═╡ 518d2d5a-219a-4a66-b035-2fd6844cebd4
y = data[:,"fruity odor"]

# ╔═╡ 86a8a3f1-475a-479e-9d4a-a646605dcd93
true_ratio = sum(y) / length(y)

# ╔═╡ 1d261a3b-fb53-4432-a50a-8887f381c433
md"## test/train split

we will follow the test/train split, then $K$-folds cross-validation procedure as outlined [here](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation). see the excellent summary figure [here](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png).

because the data set is imbalanced in terms of the labels (i.e. not a 50/50 split of fruity vs. not fruity), we will conduct a _stratified_ split, which preserves the distribution of class labels in the splits of the data.

🥝 use the `partition` function to create a 75%/25% train/test split of the data. account for the splits by storing in `ids_train` and `ids_test` the indices of the data that belong to each split. be sure to use the `shuffle` and `stratify` argument.

!!! hint
	see [here](https://alan-turing-institute.github.io/MLJ.jl/dev/preparing_data/#MLJBase.partition). also, I made a toy example to see how it works. run this a few times to understand what it's doing. make `shuffle=false` and omit `stratify=...` to understand what these arguments are doing.

	`ids_train, ids_test = partition(1:10, 0.2, 0.2, shuffle=true, stratify=vcat(ones(3), zeros(7)))`.
"

# ╔═╡ cd0e15df-270f-4403-8944-3bf1ba514ca4
ids_train, ids_test = partition(1:nrow(data), 0.75, shuffle=true,stratify=(y))

# ╔═╡ 44c1b24d-72e9-43d0-99d7-0a50182d2d9e
md"🥝 by checking the length of your `ids_train` and `ids_test` arrays, how many molecules are in the train and test sets?"

# ╔═╡ c6f72c7f-f74c-4608-a716-a7e1e7e0ae5f
begin
	println("size of ids_test - ", length(ids_test))
	println("size of ids_train - ", length(ids_train))
	(length(ids_test) + length(ids_train)) == nrow(data)
end

# ╔═╡ ea5c52d8-1e32-456c-a75e-a7a6e7987472
md"🥝 create new arrays `X_train`, `y_train`, `X_test`, and `y_test` that slice the appropriate rows of `X` and `y` so that they contain only the train/test data. "

# ╔═╡ 267cce68-79c1-4f70-baac-2ed7da51ef94
begin
	X_test = X[ids_test, :]
	y_test = y[ids_test,:]
	
	X_train = X[ids_train, :]
	y_train = y[ids_train,:]
end

# ╔═╡ 5450f7d6-f1fe-4659-90c5-af89a1a34adf
md"🥝 to check that the _stratified_ split worked correctly, what fraction of the training molecules smell fruity? what fraction of the test molecules?"

# ╔═╡ 083af0db-b7c5-4a82-9f6a-4dcf4a43df33
begin
	println("True ratio: ", true_ratio)
	println("Train ratio: ", sum(y_train) / length(y_train)) 
	println("Test ratio: ",sum(y_test) / length(y_test)) 
end

# ╔═╡ c855ccdf-215e-4537-84b1-34d5b203b1d5
md"## training a decision tree (without tuning 👀)

🥝 use the `DecisionTreeClassifier` function in scikitlearn to grow a decision tree _using the training data only_. see the [docs](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier).
"

# ╔═╡ 01b581a3-174b-4f5f-aef1-42821fbc4718
model = DecisionTreeClassifier()

# ╔═╡ 432b0872-89e1-4565-90b8-9f7f5df40294
fit!(model, X_train, y_train)

# ╔═╡ e6dac7df-a419-4ad6-8b15-eefc4299f284
md"🥝 explain what the following hyperparameters of the decision tree are doing. what are their default values?
* `max_depth`
* `criterion`
* `min_samples_split`
* `min_samples_leaf`
"

# ╔═╡ 1af62cbb-5583-4ef8-aae2-016df7683dd6
md"max_depth - the lowest leaf node level \
	criterion - The function to measure the quality of a split\
	min_sample_split - The minimum number of samples required to split an internal node \
	min_sample_leaf - The minimum number of samples required to be at a leaf node. \
	"

# ╔═╡ ddce9803-c97f-4cd1-99e8-07daabad7af7
md"🥝 what is the depth of your trained decision tree? see the [`get_depth` function](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.get_depth)."

# ╔═╡ a66e3a3e-e7ca-4564-aa8d-f757fa6f8f7c
model.get_depth()

# ╔═╡ f3793699-bfef-48b0-909a-98486a6be025
md"## evaluating the decision tree"

# ╔═╡ a543ec1f-ec35-4b00-8a52-a6cd16758f72
md"🥝 use the [`score` function](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.score) to determine the accuracy of your decision tree on the training and test data, separately. are the accuracies the same on the train and test data? why or why not? which score more likely reflects the performance of the decision tree on new, unseen data?
"

# ╔═╡ 39bf16e9-40db-45b1-a95f-e1c5c6c81117
training_score = model.score(X_train, y_train)

# ╔═╡ 2c387392-f407-4ed9-bfe2-d526de84e4ea
testing_score = model.score(X_test, y_test)

# ╔═╡ f759d732-3034-49f1-ac2e-da3fede120a8
md" The training: $training_score is pretty accurate however, this is expected giving the tree was trained on this data set. The testing score: $testing_score is less impressive and shows how our model has overfit our training data. Good start with no adjustments"

# ╔═╡ 09aa5763-328b-45b9-b7d9-371b667a0f9c
md"🥝 to make sure you understand the meaning of _accuracy_, compute the accuracy on the test data yourself by (i) using the decision tree to make predictions on the molecules in the test set using the [`predict` function](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.predict) then (ii) doing an element-wise comparison with `y_test`."

# ╔═╡ d03b2044-531d-4f41-9825-25db521165ce
begin
	y_train_predict = vec(model.predict(X_train))
	y_test_predict = vec(model.predict(X_test))
end

# ╔═╡ c75647c0-435b-4761-badf-01d5cc44765c
begin
	println(sum(y_train_predict .== y_train) / length(y_train))
	println(sum(y_test_predict .== y_test) / length(y_test))
end

# ╔═╡ 669b8c61-fe8d-41a9-9136-e08c52387e30
md"🥝 thinking of \"smells fruity\" as a \"positive\", also compute the [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) on the test set."

# ╔═╡ e6ca846a-250e-4956-866b-2db6f88a3d46
# of those predicted to be fruity, what fraction are truly fruity?

# ╔═╡ 386b034e-926d-4b53-884c-ac91bd50db7f
print("fruity? == fruity: ", sum(y_test_predict .& y_test) / sum(y_test_predict))

# ╔═╡ cc914a7f-22c6-40fa-a5eb-e42287f510bd
# of those truly fruity, what fraction are predicted fruity?

# ╔═╡ f0d34993-869e-47c9-84ae-703c8f90ec14
print("fruity == fruity?: ",sum(y_test .& y_test_predict) / sum(y_test))

# ╔═╡ 2babeadd-007c-4ccf-b28a-ac405f02a5bb
md"🥝 compute and visualize the _confusion matrix_ on the _test_ data using the [`confusion_matrix` function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html). I provide a function below to visualize the confusion matrix. make sure you understand what the confusion matrix is conveying."

# ╔═╡ 2f7c5225-e15a-4674-8f71-33f224343947
confusion_matrix(y_train, y_train_predict)

# ╔═╡ 9dbdd088-1bc7-41cb-b26c-c57dd3f1b987
function viz_confusion(cm::Matrix; odor="fruity")
	cm_to_plot = reverse(cm, dims=1)'

	fig = Figure()
	ax  = Axis(fig[1, 1], 
		xlabel="prediction", ylabel="truth",
		xticks=(1:2, ["not $odor", "$odor"]),
		yticks=(1:2, reverse(["not $odor", "$odor"]))
	)
	hm = heatmap!(cm_to_plot, 
		colormap=ColorSchemes.algae, 
		colorrange=(0, maximum(cm))
	)
	for i = 1:2
        for j = 1:2
            text!("$(Int(cm_to_plot[i, j]))",
                  position=(i, j), align=(:center, :center), color="white", 
				  textsize=50
			)
        end
    end
    Colorbar(fig[1, 2], hm, label="# molecules")
	return fig
end

# ╔═╡ b98d884e-e73e-4609-b87b-1bffdf3e4ea3
viz_confusion(confusion_matrix(y_test, y_test_predict))

# ╔═╡ 6c478b5d-a6bd-4d17-991d-54ad0390def8
viz_confusion(confusion_matrix(y_train, y_train_predict))

# ╔═╡ 549ccec9-2a45-4a61-bdeb-b92b619f2a53
md"🥝 is this performance good? how can we judge whether a machine learning model is useful/worthy or not? it is always a good idea to report the performance of a baseline model. a naive baseline model is to randomly guess whether a molecule smells fruity or not, based on the proportion of fruity molecules in the training set. what does the confusion matrix look like for this baseline model?
"

# ╔═╡ 88b38962-3a53-4519-a0df-91f6ef56ab69
md"The preforamce seems to be pretty good, It's able to identify most the samples correctly. We can judge a ML model based on the use case. If a false postive isn't too bad then we can proceed with the model no problem."

# ╔═╡ 38f88f6a-c751-4328-b4f0-f3104ba7fdd7


# ╔═╡ 96da02dd-d37a-4c26-a10e-79434c9e3d3e


# ╔═╡ b4ace69a-1f3b-45e0-b730-ddc69838f7d2


# ╔═╡ 784d1b91-aa24-4dd8-b654-230fc62c0ca6
md"## training a decision tree (with hyperparameter tuning 😎)

we now treat the `max_depth` parameter of the decision tree classifier as a tunable hyperparameter. tuning this parameter to achieve high accuracy on the test set would be cheating, known as \"data leakage\", since then our test set is being used to train the decision tree (and so isn't a proper test set anymore!). so, we resort to $K$-folds cross-validation on the training data. we expect to obtain better performance on the test set after properly tuning this hyperparameter of the decision tree.

🥝 prepare for $K=5$-fold cross validation to optimize the `max_depth` parameter. use `StratifiedCV` and `train_test_pairs` (see [here](https://docs.juliahub.com/MLJBase/jaWQl/0.18.21/resampling/#MLJBase.StratifiedCV)) to create an iterator over `(id_kf_train, id_kf_test)` tuples that give the indices of the training data that are further split into 5 rounds of train/test splits---in a way that the label distribution is preserved in the splits (hence, _stratified_).
"

# ╔═╡ fc448d06-85de-4cd6-a584-7fb70cb465d2
md"see below"

# ╔═╡ 8e6b9f25-2720-4deb-83df-558475b4f680
md"🥝 through $K=5$ folds cross-validation, let's determine the optimal `max_depth` parameter of the decision tree among $\{1, 2, ..., 20\}$. loop over each `max_depth` parameter. nested within that loop, loop over the $K=5$ folds, unpack the train/test indices (referring to the rows of `X_train` and `y_train`), train a decision tree with that `max_depth` and that chunk of training data, and compute the accuracy on that test chunk of data. ultimately, we need a length-20 array `kf_accuracy` containing the mean accuracy over the K=5 folds of test data for each `max_depth`.

* plot the mean accuracy over the 5 test folds against the value of the `max_depth`.
* what is the optimal `max_depth`?
"

# ╔═╡ 883a4e5a-02a6-4e25-87f9-a30858811fcc
begin
	itr = train_test_pairs(StratifiedCV(; nfolds=5, shuffle=true,rng=Random.GLOBAL_RNG),
		1:length(y_train), y_train)	
	
	kf_accuracy = zeros(20)
	for i in 1:20
		kf_model = DecisionTreeClassifier(max_depth=i)
		temp_kf_accuracy = 0.0
		for (f, pair) in enumerate(itr)
			kf_X_train = X[pair[1], :]
			kf_y_train = y[pair[1],:]
			kf_X_test = X[pair[2], :]
			kf_y_test = y[pair[2],:]
			fit!(kf_model, kf_X_train, kf_y_train)
			temp_kf_accuracy += kf_model.score(kf_X_test, kf_y_test)
		end
		kf_accuracy[i] = (temp_kf_accuracy / 5)
	end
end

# ╔═╡ 0be4ead7-36c2-45a3-a8d5-4e07379096a4
kf_accuracy

# ╔═╡ ad7d348e-7169-4ae4-82f8-6e39c09ab25c
begin
	local fig = Figure()
	local ax = Axis(fig[1, 1], 
		title="Average scores for k-fold testing", 
		xlabel="Max depth parameter", 
		ylabel="Accuracy")
	lines!(1:length(kf_accuracy), kf_accuracy, 
		   label="trend line", 
		   color=ColorSchemes.okabe_ito[1])
	scatter!(1:length(kf_accuracy), kf_accuracy, 
		     color=ColorSchemes.okabe_ito[2], markersize=12, 
		     label="data", marker=:rect, strokewidth=1)
	axislegend(position=:rb)
	fig
end

# ╔═╡ 9d2b2e4b-9d15-408d-b687-58f759aa8245
md"🥝 finally, train a decision tree _with the optimal max depth_, using _all_ of the training data. re-plot the confusion matrix on the _test_ data. did the performance on the test data improve?"

# ╔═╡ 1943a4aa-df4a-45bd-9804-1901fa361295
begin
	depth = argmax(kf_accuracy)
	final_model = DecisionTreeClassifier(max_depth=depth)
	fit!(final_model, X_train, y_train)
	println(final_model.score(X_test, y_test))
end

# ╔═╡ a4f2081e-342c-4884-9bf8-2f26e132223e
my_tree = final_model

# ╔═╡ ce971ec4-efdb-4c8b-a631-a2f418396562
md"🥝 one advantage of a decision tree is interpretability. you can look at each decision as a data point percolates down the tree. use `plot_tree` to visualize the decision tree. this requires matplotlib. the code below worked for me. what is the first decision in the tree? what chemistry is this split looking at (keep in mind, Python is indexed at 0)? see [here](https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/MACCSkeys.py) for a list of the SMARTS patterns involved in MAACS fingerprints. [this website](https://smarts.plus/) may be helpful to understand the pattern the SMARTS string is looking for.

```julia
import PyPlot
f = PyPlot.figure()
plot_tree(my_tree, class_names=[\"not fruity\", \"fruity\"], filled=true)
f.savefig(\"decistion_tree.pdf\", format=\"pdf\")
```
"

# ╔═╡ 93d6bae8-742c-4252-99ec-429697a19e9c
begin
	import PyPlot
	f = PyPlot.figure()
	plot_tree(my_tree, class_names=["not fruity", "fruity"], filled=true)
	f.savefig("/Users/tank/Documents/Documents/PhD/PhD_Year_2/CHE_599/decistion_tree.pdf", format="pdf")
end

# ╔═╡ 45da6eee-e99d-4345-8225-8497b3f75763
md"🥝 this data set is highly biased. the molecules were chosen due to their potential to smell pleasantly, for perfume companies. your decision tree was trained on this biased sample. do you suspect the accuracy of the decision tree to be retained if it is deployed to predict whether or not molecules in a chemistry lab on campus will smell fruity or not? to learn more about \"distribution shift\" in machine learning, see [these slides](https://docs.google.com/presentation/d/1tuCIbk9Pye-RK1xqiiZXPzT8lIgDUL6CqBkFSYZXkbY/edit).
"

# ╔═╡ 7ebd3204-6277-402d-8d3a-e67c0a91854a
md"I don't beileve the model would retain it's accuracy, the data set wasn't large enough"

# ╔═╡ Cell order:
# ╟─3b6d11a6-4bb2-492a-8bd4-24e9c7bf719b
# ╟─51633144-4549-438b-9173-29918bcda346
# ╠═c4c85f7a-4c12-11ed-266d-a71f1fde33f8
# ╠═04286557-a3e4-4858-bbc4-c99b1e88821a
# ╠═89020e90-caaa-4758-91f8-ff775af36c1b
# ╠═b99239b9-4d53-4c1e-a8a3-f6374f8e6795
# ╟─9631ebda-9ca5-4d98-b956-feba97b6bef4
# ╠═c64b517f-1351-48c5-8fab-73ab5294334e
# ╟─17f913e8-0b76-463f-8354-02744f096ab6
# ╠═eab25b4b-3ab9-4b65-89b8-f3eaf535282a
# ╠═0ae7bc0b-689e-4b34-8783-bb518bf3d7de
# ╟─793170c3-9031-415a-b592-b86735f3272e
# ╠═070ff73c-2159-419e-aeaf-76118eb92f73
# ╟─700584de-1b6c-4414-ac90-2137e5566095
# ╠═71d32941-d7f6-45bd-b347-5717c65d3b29
# ╟─fcf7c6d5-dd8c-4d9f-8c5a-c4a13f33a44a
# ╠═c2528ff3-de8b-4f3c-979a-88c5f3e21d1b
# ╠═2e906f65-6175-4681-9441-190b1f536ba0
# ╟─9d0aae10-caa5-4d44-a095-fadf5bb97d37
# ╠═a53cbbf9-0b4e-4cc1-a174-4086ecb85b63
# ╟─1ae86cb8-6da7-47ba-8c03-42bc485396b8
# ╠═b3b795ff-7439-4b74-9d67-eef24c5b49a4
# ╟─25200c84-03a2-4f71-b480-ec206976abbf
# ╠═a1246cc6-da21-4576-859c-d88c87b45cc9
# ╠═49ffc81a-f1c2-4b70-aa2d-ce67f4f62808
# ╟─df61f5aa-ff37-46c5-8c0b-a5e4ddb9ae6c
# ╠═518d2d5a-219a-4a66-b035-2fd6844cebd4
# ╠═86a8a3f1-475a-479e-9d4a-a646605dcd93
# ╟─1d261a3b-fb53-4432-a50a-8887f381c433
# ╠═cd0e15df-270f-4403-8944-3bf1ba514ca4
# ╟─44c1b24d-72e9-43d0-99d7-0a50182d2d9e
# ╠═c6f72c7f-f74c-4608-a716-a7e1e7e0ae5f
# ╟─ea5c52d8-1e32-456c-a75e-a7a6e7987472
# ╠═267cce68-79c1-4f70-baac-2ed7da51ef94
# ╟─5450f7d6-f1fe-4659-90c5-af89a1a34adf
# ╠═083af0db-b7c5-4a82-9f6a-4dcf4a43df33
# ╟─c855ccdf-215e-4537-84b1-34d5b203b1d5
# ╠═01b581a3-174b-4f5f-aef1-42821fbc4718
# ╠═432b0872-89e1-4565-90b8-9f7f5df40294
# ╟─e6dac7df-a419-4ad6-8b15-eefc4299f284
# ╟─1af62cbb-5583-4ef8-aae2-016df7683dd6
# ╟─ddce9803-c97f-4cd1-99e8-07daabad7af7
# ╠═a66e3a3e-e7ca-4564-aa8d-f757fa6f8f7c
# ╟─f3793699-bfef-48b0-909a-98486a6be025
# ╟─a543ec1f-ec35-4b00-8a52-a6cd16758f72
# ╠═39bf16e9-40db-45b1-a95f-e1c5c6c81117
# ╠═2c387392-f407-4ed9-bfe2-d526de84e4ea
# ╟─f759d732-3034-49f1-ac2e-da3fede120a8
# ╟─09aa5763-328b-45b9-b7d9-371b667a0f9c
# ╠═d03b2044-531d-4f41-9825-25db521165ce
# ╠═c75647c0-435b-4761-badf-01d5cc44765c
# ╟─669b8c61-fe8d-41a9-9136-e08c52387e30
# ╠═e6ca846a-250e-4956-866b-2db6f88a3d46
# ╠═386b034e-926d-4b53-884c-ac91bd50db7f
# ╠═cc914a7f-22c6-40fa-a5eb-e42287f510bd
# ╠═f0d34993-869e-47c9-84ae-703c8f90ec14
# ╟─2babeadd-007c-4ccf-b28a-ac405f02a5bb
# ╠═b98d884e-e73e-4609-b87b-1bffdf3e4ea3
# ╠═6c478b5d-a6bd-4d17-991d-54ad0390def8
# ╠═2f7c5225-e15a-4674-8f71-33f224343947
# ╠═9dbdd088-1bc7-41cb-b26c-c57dd3f1b987
# ╟─549ccec9-2a45-4a61-bdeb-b92b619f2a53
# ╠═88b38962-3a53-4519-a0df-91f6ef56ab69
# ╠═38f88f6a-c751-4328-b4f0-f3104ba7fdd7
# ╠═96da02dd-d37a-4c26-a10e-79434c9e3d3e
# ╠═b4ace69a-1f3b-45e0-b730-ddc69838f7d2
# ╟─784d1b91-aa24-4dd8-b654-230fc62c0ca6
# ╟─fc448d06-85de-4cd6-a584-7fb70cb465d2
# ╟─8e6b9f25-2720-4deb-83df-558475b4f680
# ╠═883a4e5a-02a6-4e25-87f9-a30858811fcc
# ╠═0be4ead7-36c2-45a3-a8d5-4e07379096a4
# ╠═ad7d348e-7169-4ae4-82f8-6e39c09ab25c
# ╟─9d2b2e4b-9d15-408d-b687-58f759aa8245
# ╠═1943a4aa-df4a-45bd-9804-1901fa361295
# ╠═a4f2081e-342c-4884-9bf8-2f26e132223e
# ╟─ce971ec4-efdb-4c8b-a631-a2f418396562
# ╠═93d6bae8-742c-4252-99ec-429697a19e9c
# ╟─45da6eee-e99d-4345-8225-8497b3f75763
# ╟─7ebd3204-6277-402d-8d3a-e67c0a91854a
