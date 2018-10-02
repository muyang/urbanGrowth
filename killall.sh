#!/bin/bash

for ((COUNTER=7023854; COUNTER<=7024008; ++COUNTER))
	do
		qdel ${COUNTER}
done