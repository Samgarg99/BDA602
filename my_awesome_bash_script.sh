#!/bin/bash
sleep 60

database_to_copy_into="baseball"
database_file="baseball.sql"


if mariadb -u root -ppassword123 -hmariadb1 ${database_to_copy_into}
then
  mariadb -u root -ppassword123 -hmariadb1 -e "use ${database_to_copy_into};"
else
  mariadb -u root -ppassword123 -hmariadb1 -e "CREATE DATABASE ${database_to_copy_into};"
  mariadb -u root -ppassword123 -hmariadb1 ${database_to_copy_into} < ${database_file}
fi

mariadb -u root -ppassword123 -hmariadb1 -e "show databases;"

mariadb -u root -ppassword123 -hmariadb1 ${database_to_copy_into} < code_baseball.sql

mariadb -u root -ppassword123 -hmariadb1 ${database_to_copy_into} -e 'select * from combined5_1;' > ./stuff/combined5_1.csv

python3 final.py

#python3 testing.py

#mariadb -u root -ppassword123 -hmariadb1 ${database_to_copy_into} -e 'select * from combined5_1;'

# mariadb -u root -ppassword123 -hmariadb1 -e "DROP DATABASE IF EXISTS ${database_to_copy_into};"

# mariadb -u root -ppassword123 -hmariadb1 -e "show databases;"

#mariadb -u root -ppassword123 -hmariadb1 ${database_to_copy_into} -e 'select * from batter3 where game_id = 12560;'

# mariadb -u root -ppassword123 -hmariadb1 -e "DROP DATABASE ${database_to_copy_into};"

#mariadb -u root -ppassword123 -hmariadb1 ${database_to_copy_into} -e 'select * from batter3;' > ./stuff/bat_avg.sql

# mariadb -u root -ppassword123 -hmariadb1 -e " select * from ${database_to_copy_into}.game where game_id = 12560;" ${database_to_copy_into}

#mariadb -u root -ppassword123 -hmariadb1 -e 'show databases;'

#mariadb -u root -ppassword123 -hmariadb1 baseball_script.sql

# mariadb -u root -ppassword123 -hmariadb1 ${database_to_copy_into} < ${database_file} }