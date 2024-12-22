import streamlit as st
import sys
import os

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from service.lyrics_generator import UserModel
from service.lyrics_classifier import Classifier


# Step 1: Perfume Description Classification
def classify_description(description):
    classifiers = {
        "gender": Classifier(feature="gender", multi_label=False, top_K=10000),
        "family": Classifier(feature="family", multi_label=False, top_K=10000),
        "subfamily": Classifier(feature="subfamily", multi_label=False, top_K=4000),
        "ingredients": Classifier(feature="ingredients", multi_label=True, top_K=200),
    }

    results = {}
    for feature, classifier in classifiers.items():
        results[feature] = classifier.classification(description)

    return results


# Step 2: Generate Complete Perfume Description
def generate_description(base_idea, name, length, features, weights):
    print(features)
    model = UserModel(base_models=features, ratios=weights, n=3)
    model.update(base_idea, ratio=1000)
    return model.generate_text(token_count=length, name=name)


def main():
    st.title("Perfume Description Helper")

    st.header("Step 1: Enter an Idea")
    name = st.text_input("Name of your perfume")
    description = st.text_area("Description of your perfume idea")

    if "classification_results" not in st.session_state:
        st.session_state["classification_results"] = {}

    if st.button("Get Suggestions for Your Perfume"):
        if description.strip():
            st.session_state["classification_results"] = classify_description(
                description
            )
        else:
            st.warning("Please enter a description idea.")

    classification_results = st.session_state["classification_results"]

    st.header("Step 2: Select Attributes")
    gender = st.pills(
        "Gender",
        ["Unisex", "Male", "Female"],
        default=(
            [classification_results.get("gender", "Unisex")]
            if classification_results
            else ["Unisex"]
        ),
    )
    family = st.pills(
        "Family",
        [
            "AMBERY",
            "AROMATIC FOUGERE",
            "CHYPRE",
            "CITRUS",
            "FLORAL",
            "LEATHER",
            "WOODY",
        ],
        default=(
            [classification_results.get("family", "AMBERY")]
            if classification_results
            else ["AMBERY"]
        ),
    )
    subfamily = st.pills(
        "Subfamily",
        [
            "ALDEHYDIC",
            "AMBERY",
            "AROMATIC FOUGERE",
            "CHYPRE",
            "CITRUS",
            "FLORAL",
            "FRUITY",
            "GOURMAND",
            "GREEN",
            "LEATHER",
            "MUSK SKIN",
            "SPICY",
            "TOBACCO",
            "WATERY",
            "WOODY",
        ],
        default=(
            [classification_results.get("subfamily", "ALDEHYDIC")]
            if classification_results
            else ["ALDEHYDIC"]
        ),
    )
    ingredients = st.pills(
        "Ingredients",
        [
            "Amber",
            "Benzoin",
            "Bergamot",
            "Cardamom",
            "Cedarwood",
            "Ciste Labdanum",
            "Geranium",
            "Grapefruit",
            "Incense - Olibanum",
            "Iris - Orris",
            "Jasmine",
            "Lavender",
            "Leather",
            "Lemon",
            "Lily Of The Valley",
            "Mandarin",
            "Musk",
            "Neroli",
            "Orange Blossom",
            "Oud - Agarwood",
            "Patchouli",
            "Pink Pepper",
            "Rose",
            "Sandalwood",
            "Tonka Bean",
            "Vanilla",
            "Vetiver",
            "Violet",
            "Woody Notes",
            "Ylang-ylang",
        ],
        default=(
            classification_results.get("ingredients", [])
            if classification_results
            else []
        ),
        selection_mode="multi",
    )

    st.header("Step 3: Generate Description")
    st.subheader("Feature Weights")
    st.caption(
        "Weights that decides how much the specified features values influence the generated description."
    )
    gender_weight = st.slider("Gender", 1, 10, 5)
    family_weight = st.slider("Family", 1, 10, 5)
    subfamily_weight = st.slider("Subfamily", 1, 10, 5)
    ingredients_weight = st.slider("Ingredients", 1, 10, 5)

    base_idea = description

    st.subheader("Generation Length")
    length = st.number_input(
        "Token Length",
        min_value=10,
        max_value=500,
        value=50,
    )

    if st.button("Generate Description"):
        features = {
            "gender": [gender],
            "family": [family],
            "subfamily": [subfamily],
            "ingredients": ingredients,
        }
        weights = {
            "gender": gender_weight,
            "family": family_weight,
            "subfamily": subfamily_weight,
            "ingredients": ingredients_weight,
        }

        print(base_idea, name, length, features, weights)
        generated_description = generate_description(
            base_idea, name, length, features, weights
        )
        st.subheader("Generated Perfume Description")
        st.text(generated_description)


if __name__ == "__main__":
    main()
