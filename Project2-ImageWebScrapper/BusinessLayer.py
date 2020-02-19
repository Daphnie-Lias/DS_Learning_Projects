#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:42:06 2020

@author: daphnie
"""

from scrapperImage.ImageWebScrapper import ImageWebScrapper

class BusinessLayer:
    
    keyword=""
    fileLoc=""
    image_name=""
    header=""
    
    
    def downloadImages( keyWord, header):
        imgScrapper = ImageWebScrapper
        url = imgScrapper.createImageUrl(keyWord)
        rawHtml = imgScrapper.scrap_html_data(url, header)
        
        imageURLList = imgScrapper.getimageUrlList(rawHtml)
        
        masterListOfImages = imgScrapper.downloadImagesFromURL(imageURLList,keyWord, header)
        
        return masterListOfImages    
   