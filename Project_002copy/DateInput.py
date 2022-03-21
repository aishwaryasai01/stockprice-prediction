import time
from datetime import datetime

# today = datetime.today().strftime('%Y-%m-%d')
# print(today)
#
# start_date = datetime.today() - datetime.timedelta(30)
#
# print(datetime.timedelta(30))
# print(start_date)

import datetime

# today = datetime.date.today()
# first = today.replace(day=1)
# lastMonth = first - datetime.timedelta(days=1)
# print(lastMonth.strftime('%Y-%m-%d'))

today = datetime.date.today()
# first = today.replace(day=1)
lastMonth = today - datetime.timedelta(days=30)

# print(lastMonth.strftime('%Y-%m-%d'))

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=30)

end_date = end_date.strftime('%Y-%m-%d')
start_date = start_date.strftime('%Y-%m-%d')
print(start_date)
print(end_date)


