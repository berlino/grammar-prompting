for ((i =1; i <= $1; i++));
do
python -u scripts/eval/retro_star_listener.py --proc_id=$i &
done
