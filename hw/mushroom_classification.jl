### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ d566a523-323b-4001-871a-1956031fd289
begin
    import Pkg
    # careful: this is _not_ a reproducible environment
    # activate the global environment
    Pkg.activate()

    using PlutoUI, CSV, DataFrames, CairoMakie, ColorSchemes
	using AlgebraOfGraphics
	import ScikitLearn
end

# ╔═╡ 26c3ac2a-e83c-445e-b841-258802724124
set_aog_theme!()

# ╔═╡ 583b698d-79ab-410f-aaeb-a09c471c83dd
TableOfContents()

# ╔═╡ f9143065-f790-4b22-bd30-0ebf8b1b8a8f
begin
	ScikitLearn.@sk_import ensemble : RandomForestClassifier
	ScikitLearn.@sk_import metrics : confusion_matrix
end

# ╔═╡ dc73c588-586f-11ed-334d-a5e71913f232
md"# mushroom classification via random forests

!!! note \"objective\"
	train and evaluate a random forest to classify mushrooms as poisonous or not, based on their attributes.

## the labeled data

🍄 download and read in the mushroom data set from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom).

!!! hint
	see the `.names` file, section 7, for the attributes. see the `header` kwarg of `CSV.read` to give the appropriate column names.
"

# ╔═╡ 2d27a20a-6699-4b18-b465-87a6080c62d8
header = ["class", "cap-shape", "cap-surface", "cap-color", "bruises?", 
	      "odor", "gill-attachment", "gill-spacing", "gill-size", 
	      "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
	      "stalk-surface-below-ring", "stalk-color-above-ring", 
	      "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
		  "ring-type", "spore-print-color", "population", "habitat"] # you're welcome

# ╔═╡ 9702bafd-6847-49bb-bba9-6524fafea360
begin
	download("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", "data.csv")
	mush_data = CSV.read("data.csv", DataFrame, header=header)
end

# ╔═╡ ecf6d038-f102-4dfb-b75b-559977e19876
md"🍄 how many features (attributes) of the mushrooms are recorded in the data set?"

# ╔═╡ 253cb90c-702c-4fa4-84ed-4c390bfbcc3f
println("Feature count: " * string(size(mush_data)[2]))

# ╔═╡ 79b8b317-a83c-4bd2-bda9-8f591150a658


# ╔═╡ 1334afa0-692a-4818-9444-bce9971a5415
md"🍄 are there any missing values in the data? if so, drop the rows that contain `missing` entries."

# ╔═╡ 8abf9193-f518-49df-8e0e-16f8e0b89d9a
#missing(data)

# ╔═╡ 48cb6573-d126-4571-8846-fc15f0ea801b
md"🍄 use `combine` and `groupby` to determine how many of the mushrooms are edible vs. not."

# ╔═╡ 8e887f73-eb02-4c3a-9b0c-1264a1be9440
combine(groupby(mush_data, "class"), nrow => "count")

# ╔═╡ d983df60-8957-4ba4-a981-e8797851da09
md"🍄 the features are categorical. how many unique categories does each feature have?

!!! hint
	I used `combine` with `All() .=> ...`.
"

# ╔═╡ fea46872-5758-4b65-8ab8-362c56cedcb1
combine(mush_data, All() .=> col -> length(unique(col)))

# ╔═╡ b640a2d6-b0f2-429d-ba36-467120b1fb2a
combine(mush_data, All() .=> col -> length(unique(col)))

# ╔═╡ c934bd07-a5f8-41d9-8a92-a2ae6459ae62


# ╔═╡ fbd0b2ff-2ed4-40d6-ad7a-b89b43208056
md"## exploring the data

🍄 my hypothesis is that odor alone can be used to distinguish between edible and poisonous mushrooms with reasonable accuracy. to test this hypothesis, draw a bar plot such that:
* each possible odor is listed on the x-axis
* there are two bars side-by-side for each odor: one representing poisonous mushrooms, the other representing edible mushrooms
* the height of the bar represents the number of mushrooms with that class label _and_ that odor
* the bars are colored differently according to class
* a legend indicates which color corresponds to which class
* the class name and odors are spelled out to be legible. e.g. instead of \"e\" we have \"edible\" in the legend; instead of \"n\" we have \"none\" as the odor label on the x-axis.

!!! hint
	you can do this manually via a double `groupby` and the `dodge` kwarg of `barplot`. however, I found it much easier to use `AlgebraOfGraphics.jl`, which shows an analogous example [here](https://aog.makie.org/stable/generated/penguins/#Styling-by-categorical-variables).

"

# ╔═╡ 8f79e473-9566-47af-9034-99085918f629
mush_frequency = data(mush_data) * frequency() * mapping(:odor)

# ╔═╡ 491a16bf-599a-4370-a2e1-17284ae96e8e
begin
	axis = (width = 500, height = 500)
	plt = mush_frequency * mapping(color = :class, dodge = :class)
	draw(plt; axis)
end

# ╔═╡ b273e7ee-625f-4da4-88bf-fa6e2cb3c912
md"## preparing the features for machine learning

❗ the `RandomForestClassifier` in scikit-learn does not a categorical variable as an input if the variable has more than two categories. for example, there are nine unique categories of odors. the random forest algorithm implementation cannot handle this. however, the algorithm _can_ handle binary features. so, we will convert each multi-category feature into a set of binary feature. for example, for the odor feature, we convert it into nine different binary indicator variables.

_old feature_: odor (values it can take on: pungent, almond, anise, none, foul, creosote, fishy, spicy, musty)

_new features encoding the same information_:
* odor_pungent (values it can take on: 0, 1)
* odor_almond (values it can take on: 0, 1)
...
* odor_musty (values it can take on: 0, 1)

🍄 create a new set of 117 binary features, as new columns in the mushrooms data frame, that encode the same attributes about the mushrooms as the original features. name these columns appropriately so that we can understand what each column means. for example, \"odor=pungent\" should be the column name of one of the new binary features.

!!! hint
	I used a double `for` loop and the `transform` function.
"

# ╔═╡ e0b93752-a20e-4cbf-b30a-b607db2dcfc6
new_mush_data = copy(mush_data)

# ╔═╡ 26c2aa55-217d-4fa4-9eaa-06b04119850a
begin
	for (i, col_name) in enumerate(names(mush_data))
		unique_values = unique(mush_data[:,col_name])
		for value in unique_values
			name = col_name * "=" * string(value)
			transform!(new_mush_data, col_name => (col-> (col .== value) ) => name)
		end
		select!(new_mush_data, Not(col_name))
	end
end

# ╔═╡ 43c76179-fb78-419c-a7c4-fbcef302b648
size(new_mush_data)

# ╔═╡ 0356bfb1-0dad-4944-9fdd-1ec8512f1bf8


# ╔═╡ 8fc789e8-3f31-46f7-932d-ae3489c8132a
md"🍄 create the (# mushrooms × # binary features) feature matrix `X` with the binarized feature vector of each mushroom in the rows.
"

# ╔═╡ f9b8c4b8-fda8-4274-b977-2231a1e6c90a
begin
	x = convert.(Float64, select(new_mush_data, Not(["class=e", "class=p"])))
	x = Matrix(x) 
end

# ╔═╡ ae66dc33-9e0b-4b3a-a3ba-65d338e7add3
md"🍄 create a # mushrooms-length target vector `y` listing the labels of the mushrooms. of course, the rows of `y` and rows of `X` must refer to the same mushroom..."

# ╔═╡ 99798d86-2309-4b31-ac8c-819a60601d3f
begin
	y = convert.(Float64, select(new_mush_data, "class=p"))
	y = Matrix(y) 
end

# ╔═╡ 54b4c02f-3c61-44f4-97d3-3e4dd6b2aea5
md"## train and evaluate the random forest classifier

🍄 grow a random forest classifier using _all_ of the data. 

!!! note
	what are the labeled input-output pairs for the random forest here? the input is the vector of binarized features representing attributes of the mushrooms. the output is the label of poisonous or edible.

ensure the random forest is set up to evaluate the predictions on the out-of-bag samples to justify _not_ using a train/test split. use the default settings of the random forest, which tend to work well out-of-the-box.

!!! hint
	see the scikit-learn docs for `RandomForestClassifier` [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_). use the `fit` method, as you did with decision trees.
"

# ╔═╡ eca30527-400d-4a70-a66a-84599b174130
begin
	clf = RandomForestClassifier(oob_score=true)
	clf.fit(x, vec(y)) 
end

# ╔═╡ 0c51f438-0bd4-4aef-aec8-b3a5ec437f41
md"🍄 how many decision trees comprise your random forest?"

# ╔═╡ 46eb3255-bb27-49ce-b478-08d3948aa1a6
clf.get_params()["n_estimators"]

# ╔═╡ c4e01a74-d669-40e2-9510-0435b53fff60
md"🍄 draw a histogram of the depths of the trees in the forest. you should see a diversity of depths, owing to the randomness of the trees.

!!! hint
	see the `estimators_` attribute, which gives a list of the individual decision trees. then like the previous assignment, use `get_depth()`.
"


# ╔═╡ e6da1914-94d3-40ac-9278-220cad1f3f2a
begin
	trees = clf.estimators_
	all_depths = zeros(length(trees))
	for (i, tree) in enumerate(trees)
		all_depths[i] = tree.get_depth()
	end
	local fig = Figure()
	local ax = Axis(fig[1, 1])
	hist!(all_depths, color="limegreen", strokecolor=:black, strokewidth=1)
	ylims!(0, nothing)
	fig
end

# ╔═╡ 905067b8-b563-4935-b551-011e6c69639e


# ╔═╡ d66d25ca-07c4-490e-a2ae-64fb49ab634d
md"🍄 write the two major ways by which random forests inject randomness into each decision tree that comprises it. this leads to the decorrelated trees.

1. each tree is trained on...
2. at each split within a tree, ...
"

# ╔═╡ dc2ac948-c629-4223-8005-7793ad8ad30d
md"🍄 compute the confusion matrix using the out of bag predictions on the mushrooms.

!!! hint
	see the `oob_decision_function_` attribute of the random forest classifier.
"

# ╔═╡ fdb01698-654f-4b57-9846-63b0c0a173cf
function viz_confusion(cm::Matrix; metr)
	cm_to_plot = reverse(cm, dims=1)

	fig = Figure()
	ax  = Axis(fig[1, 1], 
		xlabel="prediction", ylabel="truth",
		xticks=(1:2, ["not $metr", "$metr"]),
		yticks=(1:2, reverse(["not $metr", "$metr"]))
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
    Colorbar(fig[1, 2], hm, label="# of mushrooms")
	return fig
end

# ╔═╡ 9bb63252-e6f8-419b-a3b8-9c7dd04e4425
viz_confusion(confusion_matrix(y, clf.predict(x)), metr="Edible")

# ╔═╡ 3e73e288-9f28-46d1-a713-f8fce22c406c
sum(clf.predict(x) .== y) / length(y)

# ╔═╡ 3dabd5d6-ea1a-4936-a0aa-c1844912d6e5
cm = clf.oob_decision_function_

# ╔═╡ 9e5bca95-c027-447c-82f9-a21499df43d9
md"🍄 precisely explain what the \"out of bag prediction\" for a given mushroom means.

in the out-of-bag prediction for a given mushroom, we...
"

# ╔═╡ 36ab9afe-611d-464d-87f8-ba91b791fc1b
md"
## feature importance
random forests are almost always more accurate than an individual decision tree, and they are much easier to train because little tuning is needed. however, we lose interpretability because the decision of which label to place on a mushroom is being made by a committee of trees instead of just one.

🍄 compute the impurity-based importance of each feature, which is kept track of while growing the tree. draw a bar plot that shows, _for the ten most important features_:
* y-axis: the ten most important features
* x-axis: the importance score of those features
* so, bar lengths = the importance score

!!! hint
	see the `feature_importances_` attribute of the random forest classifier.
" 

# ╔═╡ 5e08c243-dbd1-4e69-8996-c4eb1f129912
names(new_mush_data)

# ╔═╡ bd65f545-1a02-4113-bb5c-3a37832f031e
begin
	feature_weights = clf.feature_importances_
	feature_names = names(new_mush_data)
	sort_ids = sortperm(feature_weights, rev=true)
	feature_weights = feature_weights[sort_ids][1:10]
	feature_names = feature_names[sort_ids][1:10]
end

# ╔═╡ 490ce163-afe6-4ed1-8a4e-0ae6a6f86a15
begin
		local fig = Figure()
		local ax = Axis(fig[1, 1],
			title="10 max weight features", 
			xlabel="features",
		 	ylabel="Weights",
			xticks=(1:length(feature_names), feature_names), 
			xticklabelrotation=45.0
		)
		barplot!(1:length(feature_weights), feature_weights, color="blue",
			strokecolor=:black, strokewidth=1,
			bar_labels=:y
		)
		ylims!(0, 0.15)
		xlims!(0, nothing)
		fig
end

# ╔═╡ Cell order:
# ╠═d566a523-323b-4001-871a-1956031fd289
# ╠═26c3ac2a-e83c-445e-b841-258802724124
# ╠═583b698d-79ab-410f-aaeb-a09c471c83dd
# ╠═f9143065-f790-4b22-bd30-0ebf8b1b8a8f
# ╟─dc73c588-586f-11ed-334d-a5e71913f232
# ╠═2d27a20a-6699-4b18-b465-87a6080c62d8
# ╠═9702bafd-6847-49bb-bba9-6524fafea360
# ╟─ecf6d038-f102-4dfb-b75b-559977e19876
# ╠═253cb90c-702c-4fa4-84ed-4c390bfbcc3f
# ╠═79b8b317-a83c-4bd2-bda9-8f591150a658
# ╟─1334afa0-692a-4818-9444-bce9971a5415
# ╠═8abf9193-f518-49df-8e0e-16f8e0b89d9a
# ╟─48cb6573-d126-4571-8846-fc15f0ea801b
# ╠═8e887f73-eb02-4c3a-9b0c-1264a1be9440
# ╟─d983df60-8957-4ba4-a981-e8797851da09
# ╠═fea46872-5758-4b65-8ab8-362c56cedcb1
# ╠═b640a2d6-b0f2-429d-ba36-467120b1fb2a
# ╠═c934bd07-a5f8-41d9-8a92-a2ae6459ae62
# ╟─fbd0b2ff-2ed4-40d6-ad7a-b89b43208056
# ╠═8f79e473-9566-47af-9034-99085918f629
# ╠═491a16bf-599a-4370-a2e1-17284ae96e8e
# ╟─b273e7ee-625f-4da4-88bf-fa6e2cb3c912
# ╠═e0b93752-a20e-4cbf-b30a-b607db2dcfc6
# ╠═26c2aa55-217d-4fa4-9eaa-06b04119850a
# ╠═43c76179-fb78-419c-a7c4-fbcef302b648
# ╠═0356bfb1-0dad-4944-9fdd-1ec8512f1bf8
# ╟─8fc789e8-3f31-46f7-932d-ae3489c8132a
# ╠═f9b8c4b8-fda8-4274-b977-2231a1e6c90a
# ╟─ae66dc33-9e0b-4b3a-a3ba-65d338e7add3
# ╠═99798d86-2309-4b31-ac8c-819a60601d3f
# ╟─54b4c02f-3c61-44f4-97d3-3e4dd6b2aea5
# ╠═eca30527-400d-4a70-a66a-84599b174130
# ╟─0c51f438-0bd4-4aef-aec8-b3a5ec437f41
# ╠═46eb3255-bb27-49ce-b478-08d3948aa1a6
# ╟─c4e01a74-d669-40e2-9510-0435b53fff60
# ╠═e6da1914-94d3-40ac-9278-220cad1f3f2a
# ╠═905067b8-b563-4935-b551-011e6c69639e
# ╟─d66d25ca-07c4-490e-a2ae-64fb49ab634d
# ╟─dc2ac948-c629-4223-8005-7793ad8ad30d
# ╠═9bb63252-e6f8-419b-a3b8-9c7dd04e4425
# ╠═fdb01698-654f-4b57-9846-63b0c0a173cf
# ╠═3e73e288-9f28-46d1-a713-f8fce22c406c
# ╠═3dabd5d6-ea1a-4936-a0aa-c1844912d6e5
# ╟─9e5bca95-c027-447c-82f9-a21499df43d9
# ╟─36ab9afe-611d-464d-87f8-ba91b791fc1b
# ╠═5e08c243-dbd1-4e69-8996-c4eb1f129912
# ╠═bd65f545-1a02-4113-bb5c-3a37832f031e
# ╠═490ce163-afe6-4ed1-8a4e-0ae6a6f86a15
