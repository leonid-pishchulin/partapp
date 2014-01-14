INCLUDEPATH += ../../../include_pb

###################################################
# protocol buffers targets
###################################################

# for some reason the joint target for h&cc is not correctly processed (it gets slash in the middle)

# this is what ideally should be generated:
#%.pb.cc %.pb.h: %.proto
#	protoc --cpp_out=./ $<

# here we add two separate targets for .h and .cc files as workaround

	

protoc_target_cc.target = %.pb.cc
protoc_target_cc.commands = ../../../external_bin/protoc -I$(<D) --cpp_out=$(<D) $<
protoc_target_cc.depends = %.proto

protoc_target_h.target = %.pb.h
protoc_target_h.commands = ../../../external_bin/protoc -I$(<D) --cpp_out=$(<D) $<
protoc_target_h.depends = %.proto

# if header files are not present qmake will not generate correct depedencies hence do:
# "qmake, make pb_touch_h, qmake, make pb_rm_h, make" when compiling for the first time

protoc_touch_h.target = pb_touch_h
protoc_touch_h.commands = touch $$PB_HEADERS
protoc_touch_h.depends = 

protoc_rm_h.target = pb_rm_h
protoc_rm_h.commands = rm -f $$PB_HEADERS
protoc_rm_h.depends = 

QMAKE_EXTRA_TARGETS += protoc_target_cc protoc_target_h protoc_touch_h protoc_rm_h


