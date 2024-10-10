# BookMate
BookMate is an application designed to streamline the process of selling books on Amazon. It utilizes the YOLOv8 algorithm to analyze photos of books to identify all the books in each image, as well as a PyTorch regression model to determine the optimal selling price. Additionally, BookMate automates the filling of Amazon-standardized spreadsheets with all necessary details.

## Installation
```bash
git clone https://github.com/DRobinson4105/bookmate.git
cd bookmate
npm install
pip install -r requirements.txt
cd training
pip install -e .
```
and install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/)

## Configuration
- Create a `.env` file in the project root directory with the following template
- Set the port number for the flask server
```sh
FLASK_PORT=port_number
```

## Running the Production Server
```bash
npm build && npm start & python api/routes.py
```

## Contributions
If you'd like to report a bug, request a feature, or contribute code, please submit an issue or pull request
