FROM --platform=linux/amd64 python:3.8.9

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
# Get necessary python libraries
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# Copy over code
COPY my_sqlcode.sql .
ADD my_awesome_bash_script.sh .

# Run app
# RUN chmod +x /app/stuff/my_awesome_bash_script.sh
CMD my_awesome_bash_script.sh
# RUN my_awesome_bash_script.sh

