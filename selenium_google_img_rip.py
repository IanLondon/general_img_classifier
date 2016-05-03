# Ian London 2016
# couldn't get anything I found to work... do it myself
# this downloads thumbnails from Google Images.

import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import base64
import re
import glob
import urllib

QUERY = 'panda'
DOWNLOAD_DIR = '%s_rip' % QUERY

target_url_str = "https://www.google.com/search?as_st=y&tbm=isch&hl=en&as_q=%s&as_epq=&as_oq=&as_eq=&cr=&as_sitesearch=&safe=images&tbs=isz:m" % QUERY

image_xpath = "//img[@class='rg_i']"

driver = webdriver.Firefox()
driver.get(target_url_str)

# function to handle dynamic page content loading - using Selenium
# modified from http://sqa.stackexchange.com/questions/3499/how-to-scroll-to-bottom-of-page-in-selenium-ide
# (thanks Polyakoff)
def scroll_down():
    # define initial page height for 'while' loop
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        time.sleep(8)
        try:
            driver.find_element_by_id('smb').click()
        except:
            print 'no See More button found...'

        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            break
        else:
            last_height = new_height


def hover(el):
    ActionChains(driver).move_to_element(el).perform()

def right_click_save_as(el):
    ActionChains(driver).move_to_element(el) \
    .context_click(el) \
    .send_keys('V') \
    .perform()

def save_img_src(el, file_no, sleep_time=0.25):
    base = el.get_attribute('src')
    # just guess jpeg, probably no file ext in url...
    file_name_full = '%s/%s.%s' % (DOWNLOAD_DIR, file_no, 'JPEG')
    try:
        urllib.urlretrieve(base, file_name_full)
        print 'wrote from url %s' % file_name_full
    except IOError as e:
        print 'Bad URL?', e

    time.sleep(sleep_time)


# Google image thumbnails are base64 html strings...
def dl_base64_img(el, file_no, sleep_time=0.25):
    hover(el)
    time.sleep(0.25)

    base = el.get_attribute('src')
    if not base:
        print 'no img', file_no
        return

    base_clean = base[base.find(','):]
    try:
        base_filetype = re.findall(r'image/(.*);', base)[0]
    except IndexError:
        print 'no img filetype... trying to save src', file_no
        save_img_src(el, file_no)
        return

    file_name_full = '%s/%s.%s' % (DOWNLOAD_DIR, file_no, base_filetype)
    with open(file_name_full, 'w') as f:
        f.write(base64.decodestring(base_clean))

    print 'wrote %s' % file_name_full
    time.sleep(sleep_time)

if __name__ == "__main__":
    # TODO: use command line args instead of hard-coded vars (eg for query)

    # scroll down to load many images
    scroll_down()

    prev_file_no = 0 #on an aborted scrape, set this to the last file written + 1
    imgs = driver.find_elements_by_xpath(image_xpath)
    # iterate thru all images, and when you're done, check to see if there are any more
    # for some reason the first image_xpath returns only 100 images,
    # so you have to keep doing find_elements_by_xpath again and again
    while prev_file_no < 1000 and len(imgs) > prev_file_no:
        scroll_down()
        imgs = driver.find_elements_by_xpath(image_xpath)
        print 'new loop. found %i images, prev_file_no was %i' % (len(imgs), prev_file_no)
        for file_no, img_el in enumerate(imgs[prev_file_no:]):
            dl_base64_img(img_el, file_no+prev_file_no)
        prev_file_no = len(imgs)
