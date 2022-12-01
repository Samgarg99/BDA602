database_to_copy_into="baseball"
database_file="baseball.sql"
mariadb -u root -p -e "CREATE DATABASE ${database_to_copy_into}"
mariadb -u root -p ${database_to_copy_into} < ${database_file}