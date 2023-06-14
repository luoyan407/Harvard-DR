#!/bin/bash
export API_TOKEN=099ea220-2608-46ea-90dd-4f7e32defa81
export SERVER_URL=https://dataverse.harvard.edu/
export PERSISTENT_ID=doi:10.7910/DVN/NCAB6P

curl -L -O -J -H "" $SERVER_URL/api/access/dataset/:persistentId/?persistentId=$PERSISTENT_ID
# curl -L -O -J -H "X-Dataverse-key:$API_TOKEN" $SERVER_URL/api/access/dataset/:persistentId/?persistentId=$PERSISTENT_ID
# curl -L -O -J -H X-Dataverse-key:$API_TOKEN $SERVER_URL/api/access/dataset/:persistentId/?persistentId=$PERSISTENT_ID