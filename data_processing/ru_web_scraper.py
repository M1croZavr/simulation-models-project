import datetime
import pandas as pd
import re
import requests

from bs4 import BeautifulSoup


def parse_date(date_string):
    """
    Parse date from web page into datetime format.
    
    Parameters
    ----------
    date_string: str
        Date string from web page.
    
    Returns
    -------
        Datetime representation of date_string.
    """
    month_name_to_number = {
        'января': 1,
        'февраля': 2,
        'марта': 3,
        'апреля': 4,
        'мая': 5,
        'июня': 6,
        'июля': 7,
        'августа': 8,
        'сентября': 9,
        'октября': 10,
        'ноября': 11,
        'декабря': 12
    }
    day_number = int(re.findall(r'^\d+', date_string)[0])
    month_name = re.findall(r'\w{3,}', date_string)[0].lower()
    month_number = month_name_to_number.get(month_name)
    year_number = int(re.findall(r'\d{4}', date_string)[0])
    return datetime.datetime(year_number, month_number, day_number)


def make_range_from_date_string(string):
    """
    Converts date strings from web page scraping into 
    list of dates range from start date to end date.
    
    Parameters
    ----------
    string: str
        Date string from web page.
    
    Returns
    -------
        List of date range.
    """
    splitted = string.split(' - ')
    from_date = string
    to_date = None
    if len(splitted) > 1:
        from_date, to_date = map(parse_date, splitted)
    if to_date is None:
        return [parse_date(from_date)]
    else:
        return [from_date + datetime.timedelta(day) for day in range((to_date - from_date).days + 1)]
    

def build_dataframe(page_content):
    """
    Builds a pd.DataFrame from page context extracted by request from https://base.garant.ru/10180094/.
    
    Parameters
    ----------
    page_content: str
        Page html markup.
    
    Returns
    -------
        Resulting pd.DataFrame of date and interest rate.
    """
    bs = BeautifulSoup(page_content, 'html.parser')
    table_block = bs.find('div', {'gtitle': 'Ключевая ставка ЦБ РФ'}).find('table')
    dates = list(map(lambda row: row.text, table_block.find_all('p', class_='s_16')))[::2]
    interest_rates = list(map(lambda row: row.text, table_block.find_all('p', class_='s_1')))

    dates_list = []
    interest_rates_list = []
    for date, interest_rate in zip(dates, interest_rates):
        date_range = make_range_from_date_string(date)
        dates_list += date_range
        interest_rates_list += [float(interest_rate.replace(',', '.'))] * len(date_range)
    df = pd.DataFrame({'date': dates_list, 'interest_rate': interest_rates_list}).sort_values('date').reset_index(drop=True)
    return df


if __name__ == '__main__':
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
    }
    url = 'https://base.garant.ru/10180094/'
    request = requests.get(url, headers=headers)
    if request.status_code == 200:
        content = request.content
        interest_rate_df = build_dataframe(content)
        interest_rate_df.to_csv('./data/ru_interest_rate.csv', index=False)
    else:
        print('Bad request', request.status_code)
