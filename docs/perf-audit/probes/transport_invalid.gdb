set pagination off
set follow-fork-mode child
set detach-on-fork on
break subscription.cpp:102
commands
silent
printf "INVALID_FRAME\n"
print hn
print n
print rndv
print f
print s->session
print s->credited
print s->freed
print s->arrived
print s->plan.ring_depth
print s->plan.payload_bytes
print s->plan.sizes
print s->rings->busy
bt 5
continue
end
run --frames 100 --bytes 75000 --batch 1 --time-ucx
