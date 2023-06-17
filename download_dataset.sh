#!/bin/bash
export SERVER_URL=https://dataverse.harvard.edu/
export PERSISTENT_ID=doi:10.7910/DVN/NCAB6P

curl -L -O -J -H "" $SERVER_URL/api/access/dataset/:persistentId/?persistentId=$PERSISTENT_ID
