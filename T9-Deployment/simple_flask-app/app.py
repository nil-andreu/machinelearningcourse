from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle # For being able to open the ML model
import numpy
import sqlite3

# We create a Hashing Vectorizer