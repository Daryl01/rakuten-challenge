# Rakuten France Multimodal Product Data Classification

## Table of content

1. [Context](#Context)
2. [Problem Description](#Problem_Description)
3. [Metric](#Metric)
4. [Usage](#Usage)
5. [Project structure](Project_structure)



## 1. Context

This challenge focuses on the topic of large-scale product type code multimodal (text and image) classification where the goal is to predict each product’s type code as defined in the catalog of Rakuten France.

The cataloging of product listings through title and image categorization is a fundamental problem for any e-commerce marketplace, with applications ranging from personalized search and recommendations to query understanding. Manual and rule-based approaches to categorization are not scalable since commercial products are organized in many classes. Deploying multimodal approaches would be a useful technique for e-commerce companies as they have trouble categorizing products given images and labels from merchants and avoid duplication, especially when selling both new and used products from professional and non-professional merchants, like Rakuten does. Advances in this area of research have been limited due to the lack of real data from actual commercial catalogs. The challenge presents several interesting research aspects due to the intrinsic noisy nature of the product labels and images, the size of modern e-commerce catalogs, and the typical unbalanced data distribution.



## 2. Problem Description
The goal of this data Project is large-scale multimodal (text and image) product data classification into product type codes.

For example, in Rakuten France catalog, a product with a French designation or title Klarstein Présentoir 2 Montres Optique Fibre associated with an image and sometimes with an additional description. This product is categorized under the 1500 product type code. There are other products with different titles, images and with possible descriptions, which are under the same product type code. Given these information on the products, like the example above, this challenge proposes to model a classifier to classify the products into its corresponding product type code.



## 3. Metric

The metric used in this challenge to rank the participants is the weighted-F1 score.

Scikit-Learn package has an F1 score implementation (link) and can be used for this challenge with its average parameter set to "weighted".



## 4. Usage
### How to set it up?



## 5. Project structure
### Notebooks
### src files
### models
### reports