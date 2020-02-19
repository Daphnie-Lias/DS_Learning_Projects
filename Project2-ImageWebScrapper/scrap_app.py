#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 06:55:56 2020

@author: daphnie
"""
# import libraries
from flask_cors import CORS,cross_origin
from flask import Flask,render_template,request,jsonify
from scrapperImage.ImageWebScrapper import ImageWebScrapper
from businesslayer.BusinessLayer import BusinessLayer
import os

# Initialize Flask app

scrap_app = Flask(__name__)


# Redirecting to home page
@scrap_app.route('/')
@cross_origin()


def home():
    return render_template('index.html')

# Redirection 
@scrap_app.route('/showImages')
@cross_origin()

def displayImages():
    list_Images = os.listdir('static')
    print(list_Images)
    
    try:
        if(len(list_Images)>0):
            return render_template('showImage.html',user_images=list_Images)
        else:
            return ("No images present")
        
    except Exception as e:
        print("No images found for this keyword",e)
        return "Please try with different search keyword"

@scrap_app.route('/searchImages',methods=['Get','Post'])
def searchImages():
    if request.method=="POST":
        search_term=request.form['keyword'] # assigning the value of the input keyword to the variable keyword
    else:
        print("Please enter something")
        
    imagescrapperutil=BusinessLayer ## Instantiate a object for ScrapperImage Class
    imagescrapper=ImageWebScrapper()
    list_images=os.listdir('static')
    imagescrapper.delete_downloaded_images(list_images)## Delete the old images before search
    
    image_name=search_term.split()
    image_name="+".join(image_name)
    
    ## We need to add the header metadata
    
    header={
        'User-Agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36"
            
            }
    lst_images=imagescrapperutil.downloadImages(search_term,header)
    
    return displayImages() # redirect the control to the show images method    
        
    
if __name__ == "__main__":
    scrap_app.run(host='127.0.0.1',port=8000)    
    