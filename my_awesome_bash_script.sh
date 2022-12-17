#!/bin/bash
sleep 10

database_to_copy_into="baseball"
database_file="baseball.sql"


# mariadb -u root -ppassword123 -hmariadb1 -e "show databases;"

if mariadb -u root -ppassword123 -hmariadb1 ${database_to_copy_into}
then
  mariadb -u root -ppassword123 -hmariadb1 -e "use ${database_to_copy_into};"
else
  mariadb -u root -ppassword123 -hmariadb1 -e "CREATE DATABASE ${database_to_copy_into};"
  mariadb -u root -ppassword123 -hmariadb1 ${database_to_copy_into} < ${database_file}
fi

mariadb -u root -ppassword123 -hmariadb1 -e "show databases;"

mariadb -u root -ppassword123 -hmariadb1 ${database_to_copy_into} < baseball_code.sql

mariadb -u root -ppassword123 -hmariadb1 ${database_to_copy_into} -e 'select * from batter3 where game_id = 12560;'

mariadb -u root -ppassword123 -hmariadb1 ${database_to_copy_into} -e 'select * from batter3;' > ./stuff/bat_avg.sql

# mariadb -u root -ppassword123 -hmariadb1 -e " select * from ${database_to_copy_into}.game where game_id = 12560;" ${database_to_copy_into}

#mariadb -u root -ppassword123 -hmariadb1 -e 'show databases;'

#mariadb -u root -ppassword123 -hmariadb1 baseball_script.sql

# mariadb -u root -ppassword123 -hmariadb1 ${database_to_copy_into} < ${database_file} }