��	
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.02unknown8��
�
conv1d_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameconv1d_80/kernel
y
$conv1d_80/kernel/Read/ReadVariableOpReadVariableOpconv1d_80/kernel*"
_output_shapes
:
*
dtype0
t
conv1d_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_80/bias
m
"conv1d_80/bias/Read/ReadVariableOpReadVariableOpconv1d_80/bias*
_output_shapes
:*
dtype0
�
conv1d_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1d_81/kernel
y
$conv1d_81/kernel/Read/ReadVariableOpReadVariableOpconv1d_81/kernel*"
_output_shapes
:*
dtype0
t
conv1d_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_81/bias
m
"conv1d_81/bias/Read/ReadVariableOpReadVariableOpconv1d_81/bias*
_output_shapes
:*
dtype0
z
dense_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_80/kernel
s
#dense_80/kernel/Read/ReadVariableOpReadVariableOpdense_80/kernel*
_output_shapes

:
*
dtype0
r
dense_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_80/bias
k
!dense_80/bias/Read/ReadVariableOpReadVariableOpdense_80/bias*
_output_shapes
:
*
dtype0
z
dense_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_81/kernel
s
#dense_81/kernel/Read/ReadVariableOpReadVariableOpdense_81/kernel*
_output_shapes

:
*
dtype0
r
dense_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_81/bias
k
!dense_81/bias/Read/ReadVariableOpReadVariableOpdense_81/bias*
_output_shapes
:*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Nadam/conv1d_80/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameNadam/conv1d_80/kernel/m
�
,Nadam/conv1d_80/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_80/kernel/m*"
_output_shapes
:
*
dtype0
�
Nadam/conv1d_80/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameNadam/conv1d_80/bias/m
}
*Nadam/conv1d_80/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_80/bias/m*
_output_shapes
:*
dtype0
�
Nadam/conv1d_81/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameNadam/conv1d_81/kernel/m
�
,Nadam/conv1d_81/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_81/kernel/m*"
_output_shapes
:*
dtype0
�
Nadam/conv1d_81/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameNadam/conv1d_81/bias/m
}
*Nadam/conv1d_81/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_81/bias/m*
_output_shapes
:*
dtype0
�
Nadam/dense_80/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_80/kernel/m
�
+Nadam/dense_80/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_80/kernel/m*
_output_shapes

:
*
dtype0
�
Nadam/dense_80/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_80/bias/m
{
)Nadam/dense_80/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_80/bias/m*
_output_shapes
:
*
dtype0
�
Nadam/dense_81/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_81/kernel/m
�
+Nadam/dense_81/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_81/kernel/m*
_output_shapes

:
*
dtype0
�
Nadam/dense_81/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_81/bias/m
{
)Nadam/dense_81/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_81/bias/m*
_output_shapes
:*
dtype0
�
Nadam/conv1d_80/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameNadam/conv1d_80/kernel/v
�
,Nadam/conv1d_80/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_80/kernel/v*"
_output_shapes
:
*
dtype0
�
Nadam/conv1d_80/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameNadam/conv1d_80/bias/v
}
*Nadam/conv1d_80/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_80/bias/v*
_output_shapes
:*
dtype0
�
Nadam/conv1d_81/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameNadam/conv1d_81/kernel/v
�
,Nadam/conv1d_81/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_81/kernel/v*"
_output_shapes
:*
dtype0
�
Nadam/conv1d_81/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameNadam/conv1d_81/bias/v
}
*Nadam/conv1d_81/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_81/bias/v*
_output_shapes
:*
dtype0
�
Nadam/dense_80/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_80/kernel/v
�
+Nadam/dense_80/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_80/kernel/v*
_output_shapes

:
*
dtype0
�
Nadam/dense_80/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_80/bias/v
{
)Nadam/dense_80/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_80/bias/v*
_output_shapes
:
*
dtype0
�
Nadam/dense_81/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_81/kernel/v
�
+Nadam/dense_81/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_81/kernel/v*
_output_shapes

:
*
dtype0
�
Nadam/dense_81/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_81/bias/v
{
)Nadam/dense_81/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_81/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�6
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�5
value�5B�5 B�5
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
	trainable_variables

	variables
regularization_losses
	keras_api

signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
 regularization_losses
!	keras_api
R
"trainable_variables
#	variables
$regularization_losses
%	keras_api
h

&kernel
'bias
(trainable_variables
)	variables
*regularization_losses
+	keras_api
h

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
�
2iter

3beta_1

4beta_2
	5decay
6learning_rate
7momentum_cachemkmlmmmn&mo'mp,mq-mrvsvtvuvv&vw'vx,vy-vz
8
0
1
2
3
&4
'5
,6
-7
8
0
1
2
3
&4
'5
,6
-7
 
�
8non_trainable_variables
9layer_metrics
	trainable_variables

	variables

:layers
;metrics
regularization_losses
<layer_regularization_losses
 
\Z
VARIABLE_VALUEconv1d_80/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_80/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
=layer_metrics
>non_trainable_variables
trainable_variables
	variables

?layers
@metrics
regularization_losses
Alayer_regularization_losses
 
 
 
�
Blayer_metrics
Cnon_trainable_variables
trainable_variables
	variables

Dlayers
Emetrics
regularization_losses
Flayer_regularization_losses
\Z
VARIABLE_VALUEconv1d_81/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_81/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Glayer_metrics
Hnon_trainable_variables
trainable_variables
	variables

Ilayers
Jmetrics
regularization_losses
Klayer_regularization_losses
 
 
 
�
Llayer_metrics
Mnon_trainable_variables
trainable_variables
	variables

Nlayers
Ometrics
 regularization_losses
Player_regularization_losses
 
 
 
�
Qlayer_metrics
Rnon_trainable_variables
"trainable_variables
#	variables

Slayers
Tmetrics
$regularization_losses
Ulayer_regularization_losses
[Y
VARIABLE_VALUEdense_80/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_80/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1

&0
'1
 
�
Vlayer_metrics
Wnon_trainable_variables
(trainable_variables
)	variables

Xlayers
Ymetrics
*regularization_losses
Zlayer_regularization_losses
[Y
VARIABLE_VALUEdense_81/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_81/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 
�
[layer_metrics
\non_trainable_variables
.trainable_variables
/	variables

]layers
^metrics
0regularization_losses
_layer_regularization_losses
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE
 
 
1
0
1
2
3
4
5
6

`0
a1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	btotal
	ccount
d	variables
e	keras_api
D
	ftotal
	gcount
h
_fn_kwargs
i	variables
j	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

b0
c1

d	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

f0
g1

i	variables
�~
VARIABLE_VALUENadam/conv1d_80/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/conv1d_80/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUENadam/conv1d_81/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/conv1d_81/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_80/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_80/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_81/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_81/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUENadam/conv1d_80/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/conv1d_80/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUENadam/conv1d_81/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/conv1d_81/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_80/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_80/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_81/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_81/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_conv1d_80_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_80_inputconv1d_80/kernelconv1d_80/biasconv1d_81/kernelconv1d_81/biasdense_80/kerneldense_80/biasdense_81/kerneldense_81/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_5984850
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv1d_80/kernel/Read/ReadVariableOp"conv1d_80/bias/Read/ReadVariableOp$conv1d_81/kernel/Read/ReadVariableOp"conv1d_81/bias/Read/ReadVariableOp#dense_80/kernel/Read/ReadVariableOp!dense_80/bias/Read/ReadVariableOp#dense_81/kernel/Read/ReadVariableOp!dense_81/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Nadam/conv1d_80/kernel/m/Read/ReadVariableOp*Nadam/conv1d_80/bias/m/Read/ReadVariableOp,Nadam/conv1d_81/kernel/m/Read/ReadVariableOp*Nadam/conv1d_81/bias/m/Read/ReadVariableOp+Nadam/dense_80/kernel/m/Read/ReadVariableOp)Nadam/dense_80/bias/m/Read/ReadVariableOp+Nadam/dense_81/kernel/m/Read/ReadVariableOp)Nadam/dense_81/bias/m/Read/ReadVariableOp,Nadam/conv1d_80/kernel/v/Read/ReadVariableOp*Nadam/conv1d_80/bias/v/Read/ReadVariableOp,Nadam/conv1d_81/kernel/v/Read/ReadVariableOp*Nadam/conv1d_81/bias/v/Read/ReadVariableOp+Nadam/dense_80/kernel/v/Read/ReadVariableOp)Nadam/dense_80/bias/v/Read/ReadVariableOp+Nadam/dense_81/kernel/v/Read/ReadVariableOp)Nadam/dense_81/bias/v/Read/ReadVariableOpConst*/
Tin(
&2$	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_5985228
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_80/kernelconv1d_80/biasconv1d_81/kernelconv1d_81/biasdense_80/kerneldense_80/biasdense_81/kerneldense_81/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcounttotal_1count_1Nadam/conv1d_80/kernel/mNadam/conv1d_80/bias/mNadam/conv1d_81/kernel/mNadam/conv1d_81/bias/mNadam/dense_80/kernel/mNadam/dense_80/bias/mNadam/dense_81/kernel/mNadam/dense_81/bias/mNadam/conv1d_80/kernel/vNadam/conv1d_80/bias/vNadam/conv1d_81/kernel/vNadam/conv1d_81/bias/vNadam/dense_80/kernel/vNadam/dense_80/bias/vNadam/dense_81/kernel/vNadam/dense_81/bias/v*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_5985340��
��
�
#__inference__traced_restore_5985340
file_prefix%
!assignvariableop_conv1d_80_kernel%
!assignvariableop_1_conv1d_80_bias'
#assignvariableop_2_conv1d_81_kernel%
!assignvariableop_3_conv1d_81_bias&
"assignvariableop_4_dense_80_kernel$
 assignvariableop_5_dense_80_bias&
"assignvariableop_6_dense_81_kernel$
 assignvariableop_7_dense_81_bias!
assignvariableop_8_nadam_iter#
assignvariableop_9_nadam_beta_1$
 assignvariableop_10_nadam_beta_2#
assignvariableop_11_nadam_decay+
'assignvariableop_12_nadam_learning_rate,
(assignvariableop_13_nadam_momentum_cache
assignvariableop_14_total
assignvariableop_15_count
assignvariableop_16_total_1
assignvariableop_17_count_10
,assignvariableop_18_nadam_conv1d_80_kernel_m.
*assignvariableop_19_nadam_conv1d_80_bias_m0
,assignvariableop_20_nadam_conv1d_81_kernel_m.
*assignvariableop_21_nadam_conv1d_81_bias_m/
+assignvariableop_22_nadam_dense_80_kernel_m-
)assignvariableop_23_nadam_dense_80_bias_m/
+assignvariableop_24_nadam_dense_81_kernel_m-
)assignvariableop_25_nadam_dense_81_bias_m0
,assignvariableop_26_nadam_conv1d_80_kernel_v.
*assignvariableop_27_nadam_conv1d_80_bias_v0
,assignvariableop_28_nadam_conv1d_81_kernel_v.
*assignvariableop_29_nadam_conv1d_81_bias_v/
+assignvariableop_30_nadam_dense_80_kernel_v-
)assignvariableop_31_nadam_dense_80_bias_v/
+assignvariableop_32_nadam_dense_81_kernel_v-
)assignvariableop_33_nadam_dense_81_bias_v
identity_35��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_conv1d_80_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv1d_80_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv1d_81_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv1d_81_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_80_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_80_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_81_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_81_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_nadam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_nadam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp assignvariableop_10_nadam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_nadam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp'assignvariableop_12_nadam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp(assignvariableop_13_nadam_momentum_cacheIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp,assignvariableop_18_nadam_conv1d_80_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_nadam_conv1d_80_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp,assignvariableop_20_nadam_conv1d_81_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_nadam_conv1d_81_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp+assignvariableop_22_nadam_dense_80_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_nadam_dense_80_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp+assignvariableop_24_nadam_dense_81_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_nadam_dense_81_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp,assignvariableop_26_nadam_conv1d_80_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_nadam_conv1d_80_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp,assignvariableop_28_nadam_conv1d_81_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_nadam_conv1d_81_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp+assignvariableop_30_nadam_dense_80_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_nadam_dense_80_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp+assignvariableop_32_nadam_dense_81_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_nadam_dense_81_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_339
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_34�
Identity_35IdentityIdentity_34:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_35"#
identity_35Identity_35:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
� 
�
J__inference_sequential_40_layer_call_and_return_conditional_losses_5984695
conv1d_80_input
conv1d_80_5984586
conv1d_80_5984588
conv1d_81_5984620
conv1d_81_5984622
dense_80_5984662
dense_80_5984664
dense_81_5984689
dense_81_5984691
identity��!conv1d_80/StatefulPartitionedCall�!conv1d_81/StatefulPartitionedCall� dense_80/StatefulPartitionedCall� dense_81/StatefulPartitionedCall�
!conv1d_80/StatefulPartitionedCallStatefulPartitionedCallconv1d_80_inputconv1d_80_5984586conv1d_80_5984588*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv1d_80_layer_call_and_return_conditional_losses_59845752#
!conv1d_80/StatefulPartitionedCall�
 max_pooling1d_80/PartitionedCallPartitionedCall*conv1d_80/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling1d_80_layer_call_and_return_conditional_losses_59845332"
 max_pooling1d_80/PartitionedCall�
!conv1d_81/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_80/PartitionedCall:output:0conv1d_81_5984620conv1d_81_5984622*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv1d_81_layer_call_and_return_conditional_losses_59846092#
!conv1d_81/StatefulPartitionedCall�
 max_pooling1d_81/PartitionedCallPartitionedCall*conv1d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling1d_81_layer_call_and_return_conditional_losses_59845482"
 max_pooling1d_81/PartitionedCall�
flatten_40/PartitionedCallPartitionedCall)max_pooling1d_81/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_40_layer_call_and_return_conditional_losses_59846322
flatten_40/PartitionedCall�
 dense_80/StatefulPartitionedCallStatefulPartitionedCall#flatten_40/PartitionedCall:output:0dense_80_5984662dense_80_5984664*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_80_layer_call_and_return_conditional_losses_59846512"
 dense_80/StatefulPartitionedCall�
 dense_81/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0dense_81_5984689dense_81_5984691*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_81_layer_call_and_return_conditional_losses_59846782"
 dense_81/StatefulPartitionedCall�
IdentityIdentity)dense_81/StatefulPartitionedCall:output:0"^conv1d_80/StatefulPartitionedCall"^conv1d_81/StatefulPartitionedCall!^dense_80/StatefulPartitionedCall!^dense_81/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::2F
!conv1d_80/StatefulPartitionedCall!conv1d_80/StatefulPartitionedCall2F
!conv1d_81/StatefulPartitionedCall!conv1d_81/StatefulPartitionedCall2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_nameconv1d_80_input
�
�
/__inference_sequential_40_layer_call_fn_5984771
conv1d_80_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv1d_80_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_40_layer_call_and_return_conditional_losses_59847522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_nameconv1d_80_input
�
�
F__inference_conv1d_80_layer_call_and_return_conditional_losses_5984575

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        	               2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:���������2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_sequential_40_layer_call_fn_5985000

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_40_layer_call_and_return_conditional_losses_59848002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�
J__inference_sequential_40_layer_call_and_return_conditional_losses_5984958

inputs9
5conv1d_80_conv1d_expanddims_1_readvariableop_resource-
)conv1d_80_biasadd_readvariableop_resource9
5conv1d_81_conv1d_expanddims_1_readvariableop_resource-
)conv1d_81_biasadd_readvariableop_resource+
'dense_80_matmul_readvariableop_resource,
(dense_80_biasadd_readvariableop_resource+
'dense_81_matmul_readvariableop_resource,
(dense_81_biasadd_readvariableop_resource
identity��
conv1d_80/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        	               2
conv1d_80/Pad/paddings�
conv1d_80/PadPadinputsconv1d_80/Pad/paddings:output:0*
T0*+
_output_shapes
:���������2
conv1d_80/Pad�
conv1d_80/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
conv1d_80/conv1d/ExpandDims/dim�
conv1d_80/conv1d/ExpandDims
ExpandDimsconv1d_80/Pad:output:0(conv1d_80/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d_80/conv1d/ExpandDims�
,conv1d_80/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_80_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02.
,conv1d_80/conv1d/ExpandDims_1/ReadVariableOp�
!conv1d_80/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_80/conv1d/ExpandDims_1/dim�
conv1d_80/conv1d/ExpandDims_1
ExpandDims4conv1d_80/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_80/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d_80/conv1d/ExpandDims_1�
conv1d_80/conv1dConv2D$conv1d_80/conv1d/ExpandDims:output:0&conv1d_80/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv1d_80/conv1d�
conv1d_80/conv1d/SqueezeSqueezeconv1d_80/conv1d:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������2
conv1d_80/conv1d/Squeeze�
 conv1d_80/BiasAdd/ReadVariableOpReadVariableOp)conv1d_80_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_80/BiasAdd/ReadVariableOp�
conv1d_80/BiasAddBiasAdd!conv1d_80/conv1d/Squeeze:output:0(conv1d_80/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2
conv1d_80/BiasAdd�
max_pooling1d_80/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_80/ExpandDims/dim�
max_pooling1d_80/ExpandDims
ExpandDimsconv1d_80/BiasAdd:output:0(max_pooling1d_80/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
max_pooling1d_80/ExpandDims�
max_pooling1d_80/MaxPoolMaxPool$max_pooling1d_80/ExpandDims:output:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
max_pooling1d_80/MaxPool�
max_pooling1d_80/SqueezeSqueeze!max_pooling1d_80/MaxPool:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims
2
max_pooling1d_80/Squeeze�
conv1d_81/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_81/Pad/paddings�
conv1d_81/PadPad!max_pooling1d_80/Squeeze:output:0conv1d_81/Pad/paddings:output:0*
T0*+
_output_shapes
:���������2
conv1d_81/Pad�
conv1d_81/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
conv1d_81/conv1d/ExpandDims/dim�
conv1d_81/conv1d/ExpandDims
ExpandDimsconv1d_81/Pad:output:0(conv1d_81/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d_81/conv1d/ExpandDims�
,conv1d_81/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_81_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_81/conv1d/ExpandDims_1/ReadVariableOp�
!conv1d_81/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_81/conv1d/ExpandDims_1/dim�
conv1d_81/conv1d/ExpandDims_1
ExpandDims4conv1d_81/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_81/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_81/conv1d/ExpandDims_1�
conv1d_81/conv1dConv2D$conv1d_81/conv1d/ExpandDims:output:0&conv1d_81/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv1d_81/conv1d�
conv1d_81/conv1d/SqueezeSqueezeconv1d_81/conv1d:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������2
conv1d_81/conv1d/Squeeze�
 conv1d_81/BiasAdd/ReadVariableOpReadVariableOp)conv1d_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_81/BiasAdd/ReadVariableOp�
conv1d_81/BiasAddBiasAdd!conv1d_81/conv1d/Squeeze:output:0(conv1d_81/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2
conv1d_81/BiasAdd�
max_pooling1d_81/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_81/ExpandDims/dim�
max_pooling1d_81/ExpandDims
ExpandDimsconv1d_81/BiasAdd:output:0(max_pooling1d_81/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
max_pooling1d_81/ExpandDims�
max_pooling1d_81/MaxPoolMaxPool$max_pooling1d_81/ExpandDims:output:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
max_pooling1d_81/MaxPool�
max_pooling1d_81/SqueezeSqueeze!max_pooling1d_81/MaxPool:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims
2
max_pooling1d_81/Squeezeu
flatten_40/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_40/Const�
flatten_40/ReshapeReshape!max_pooling1d_81/Squeeze:output:0flatten_40/Const:output:0*
T0*'
_output_shapes
:���������2
flatten_40/Reshape�
dense_80/MatMul/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_80/MatMul/ReadVariableOp�
dense_80/MatMulMatMulflatten_40/Reshape:output:0&dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_80/MatMul�
dense_80/BiasAdd/ReadVariableOpReadVariableOp(dense_80_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_80/BiasAdd/ReadVariableOp�
dense_80/BiasAddBiasAdddense_80/MatMul:product:0'dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_80/BiasAdds
dense_80/ReluReludense_80/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_80/Relu�
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_81/MatMul/ReadVariableOp�
dense_81/MatMulMatMuldense_80/Relu:activations:0&dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_81/MatMul�
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_81/BiasAdd/ReadVariableOp�
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_81/BiasAddp
dense_81/EluEludense_81/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_81/Elun
IdentityIdentitydense_81/Elu:activations:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������:::::::::S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_conv1d_80_layer_call_and_return_conditional_losses_5985017

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        	               2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:���������2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_conv1d_80_layer_call_fn_5985026

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv1d_80_layer_call_and_return_conditional_losses_59845752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�J
�
 __inference__traced_save_5985228
file_prefix/
+savev2_conv1d_80_kernel_read_readvariableop-
)savev2_conv1d_80_bias_read_readvariableop/
+savev2_conv1d_81_kernel_read_readvariableop-
)savev2_conv1d_81_bias_read_readvariableop.
*savev2_dense_80_kernel_read_readvariableop,
(savev2_dense_80_bias_read_readvariableop.
*savev2_dense_81_kernel_read_readvariableop,
(savev2_dense_81_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_nadam_conv1d_80_kernel_m_read_readvariableop5
1savev2_nadam_conv1d_80_bias_m_read_readvariableop7
3savev2_nadam_conv1d_81_kernel_m_read_readvariableop5
1savev2_nadam_conv1d_81_bias_m_read_readvariableop6
2savev2_nadam_dense_80_kernel_m_read_readvariableop4
0savev2_nadam_dense_80_bias_m_read_readvariableop6
2savev2_nadam_dense_81_kernel_m_read_readvariableop4
0savev2_nadam_dense_81_bias_m_read_readvariableop7
3savev2_nadam_conv1d_80_kernel_v_read_readvariableop5
1savev2_nadam_conv1d_80_bias_v_read_readvariableop7
3savev2_nadam_conv1d_81_kernel_v_read_readvariableop5
1savev2_nadam_conv1d_81_bias_v_read_readvariableop6
2savev2_nadam_dense_80_kernel_v_read_readvariableop4
0savev2_nadam_dense_80_bias_v_read_readvariableop6
2savev2_nadam_dense_81_kernel_v_read_readvariableop4
0savev2_nadam_dense_81_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_5011cfa4a01b44a6958617864a080534/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv1d_80_kernel_read_readvariableop)savev2_conv1d_80_bias_read_readvariableop+savev2_conv1d_81_kernel_read_readvariableop)savev2_conv1d_81_bias_read_readvariableop*savev2_dense_80_kernel_read_readvariableop(savev2_dense_80_bias_read_readvariableop*savev2_dense_81_kernel_read_readvariableop(savev2_dense_81_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_nadam_conv1d_80_kernel_m_read_readvariableop1savev2_nadam_conv1d_80_bias_m_read_readvariableop3savev2_nadam_conv1d_81_kernel_m_read_readvariableop1savev2_nadam_conv1d_81_bias_m_read_readvariableop2savev2_nadam_dense_80_kernel_m_read_readvariableop0savev2_nadam_dense_80_bias_m_read_readvariableop2savev2_nadam_dense_81_kernel_m_read_readvariableop0savev2_nadam_dense_81_bias_m_read_readvariableop3savev2_nadam_conv1d_80_kernel_v_read_readvariableop1savev2_nadam_conv1d_80_bias_v_read_readvariableop3savev2_nadam_conv1d_81_kernel_v_read_readvariableop1savev2_nadam_conv1d_81_bias_v_read_readvariableop2savev2_nadam_dense_80_kernel_v_read_readvariableop0savev2_nadam_dense_80_bias_v_read_readvariableop2savev2_nadam_dense_81_kernel_v_read_readvariableop0savev2_nadam_dense_81_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
::::
:
:
:: : : : : : : : : : :
::::
:
:
::
::::
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:
: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:
: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::($
"
_output_shapes
:
: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:
:  

_output_shapes
:
:$! 

_output_shapes

:
: "

_output_shapes
::#

_output_shapes
: 
� 
�
J__inference_sequential_40_layer_call_and_return_conditional_losses_5984800

inputs
conv1d_80_5984776
conv1d_80_5984778
conv1d_81_5984782
conv1d_81_5984784
dense_80_5984789
dense_80_5984791
dense_81_5984794
dense_81_5984796
identity��!conv1d_80/StatefulPartitionedCall�!conv1d_81/StatefulPartitionedCall� dense_80/StatefulPartitionedCall� dense_81/StatefulPartitionedCall�
!conv1d_80/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_80_5984776conv1d_80_5984778*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv1d_80_layer_call_and_return_conditional_losses_59845752#
!conv1d_80/StatefulPartitionedCall�
 max_pooling1d_80/PartitionedCallPartitionedCall*conv1d_80/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling1d_80_layer_call_and_return_conditional_losses_59845332"
 max_pooling1d_80/PartitionedCall�
!conv1d_81/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_80/PartitionedCall:output:0conv1d_81_5984782conv1d_81_5984784*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv1d_81_layer_call_and_return_conditional_losses_59846092#
!conv1d_81/StatefulPartitionedCall�
 max_pooling1d_81/PartitionedCallPartitionedCall*conv1d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling1d_81_layer_call_and_return_conditional_losses_59845482"
 max_pooling1d_81/PartitionedCall�
flatten_40/PartitionedCallPartitionedCall)max_pooling1d_81/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_40_layer_call_and_return_conditional_losses_59846322
flatten_40/PartitionedCall�
 dense_80/StatefulPartitionedCallStatefulPartitionedCall#flatten_40/PartitionedCall:output:0dense_80_5984789dense_80_5984791*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_80_layer_call_and_return_conditional_losses_59846512"
 dense_80/StatefulPartitionedCall�
 dense_81/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0dense_81_5984794dense_81_5984796*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_81_layer_call_and_return_conditional_losses_59846782"
 dense_81/StatefulPartitionedCall�
IdentityIdentity)dense_81/StatefulPartitionedCall:output:0"^conv1d_80/StatefulPartitionedCall"^conv1d_81/StatefulPartitionedCall!^dense_80/StatefulPartitionedCall!^dense_81/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::2F
!conv1d_80/StatefulPartitionedCall!conv1d_80/StatefulPartitionedCall2F
!conv1d_81/StatefulPartitionedCall!conv1d_81/StatefulPartitionedCall2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_80_layer_call_and_return_conditional_losses_5985074

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
G__inference_flatten_40_layer_call_and_return_conditional_losses_5985058

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
� 
�
J__inference_sequential_40_layer_call_and_return_conditional_losses_5984722
conv1d_80_input
conv1d_80_5984698
conv1d_80_5984700
conv1d_81_5984704
conv1d_81_5984706
dense_80_5984711
dense_80_5984713
dense_81_5984716
dense_81_5984718
identity��!conv1d_80/StatefulPartitionedCall�!conv1d_81/StatefulPartitionedCall� dense_80/StatefulPartitionedCall� dense_81/StatefulPartitionedCall�
!conv1d_80/StatefulPartitionedCallStatefulPartitionedCallconv1d_80_inputconv1d_80_5984698conv1d_80_5984700*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv1d_80_layer_call_and_return_conditional_losses_59845752#
!conv1d_80/StatefulPartitionedCall�
 max_pooling1d_80/PartitionedCallPartitionedCall*conv1d_80/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling1d_80_layer_call_and_return_conditional_losses_59845332"
 max_pooling1d_80/PartitionedCall�
!conv1d_81/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_80/PartitionedCall:output:0conv1d_81_5984704conv1d_81_5984706*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv1d_81_layer_call_and_return_conditional_losses_59846092#
!conv1d_81/StatefulPartitionedCall�
 max_pooling1d_81/PartitionedCallPartitionedCall*conv1d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling1d_81_layer_call_and_return_conditional_losses_59845482"
 max_pooling1d_81/PartitionedCall�
flatten_40/PartitionedCallPartitionedCall)max_pooling1d_81/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_40_layer_call_and_return_conditional_losses_59846322
flatten_40/PartitionedCall�
 dense_80/StatefulPartitionedCallStatefulPartitionedCall#flatten_40/PartitionedCall:output:0dense_80_5984711dense_80_5984713*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_80_layer_call_and_return_conditional_losses_59846512"
 dense_80/StatefulPartitionedCall�
 dense_81/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0dense_81_5984716dense_81_5984718*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_81_layer_call_and_return_conditional_losses_59846782"
 dense_81/StatefulPartitionedCall�
IdentityIdentity)dense_81/StatefulPartitionedCall:output:0"^conv1d_80/StatefulPartitionedCall"^conv1d_81/StatefulPartitionedCall!^dense_80/StatefulPartitionedCall!^dense_81/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::2F
!conv1d_80/StatefulPartitionedCall!conv1d_80/StatefulPartitionedCall2F
!conv1d_81/StatefulPartitionedCall!conv1d_81/StatefulPartitionedCall2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_nameconv1d_80_input
�N
�
"__inference__wrapped_model_5984524
conv1d_80_inputG
Csequential_40_conv1d_80_conv1d_expanddims_1_readvariableop_resource;
7sequential_40_conv1d_80_biasadd_readvariableop_resourceG
Csequential_40_conv1d_81_conv1d_expanddims_1_readvariableop_resource;
7sequential_40_conv1d_81_biasadd_readvariableop_resource9
5sequential_40_dense_80_matmul_readvariableop_resource:
6sequential_40_dense_80_biasadd_readvariableop_resource9
5sequential_40_dense_81_matmul_readvariableop_resource:
6sequential_40_dense_81_biasadd_readvariableop_resource
identity��
$sequential_40/conv1d_80/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        	               2&
$sequential_40/conv1d_80/Pad/paddings�
sequential_40/conv1d_80/PadPadconv1d_80_input-sequential_40/conv1d_80/Pad/paddings:output:0*
T0*+
_output_shapes
:���������2
sequential_40/conv1d_80/Pad�
-sequential_40/conv1d_80/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2/
-sequential_40/conv1d_80/conv1d/ExpandDims/dim�
)sequential_40/conv1d_80/conv1d/ExpandDims
ExpandDims$sequential_40/conv1d_80/Pad:output:06sequential_40/conv1d_80/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2+
)sequential_40/conv1d_80/conv1d/ExpandDims�
:sequential_40/conv1d_80/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_40_conv1d_80_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02<
:sequential_40/conv1d_80/conv1d/ExpandDims_1/ReadVariableOp�
/sequential_40/conv1d_80/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_40/conv1d_80/conv1d/ExpandDims_1/dim�
+sequential_40/conv1d_80/conv1d/ExpandDims_1
ExpandDimsBsequential_40/conv1d_80/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_40/conv1d_80/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2-
+sequential_40/conv1d_80/conv1d/ExpandDims_1�
sequential_40/conv1d_80/conv1dConv2D2sequential_40/conv1d_80/conv1d/ExpandDims:output:04sequential_40/conv1d_80/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2 
sequential_40/conv1d_80/conv1d�
&sequential_40/conv1d_80/conv1d/SqueezeSqueeze'sequential_40/conv1d_80/conv1d:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������2(
&sequential_40/conv1d_80/conv1d/Squeeze�
.sequential_40/conv1d_80/BiasAdd/ReadVariableOpReadVariableOp7sequential_40_conv1d_80_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_40/conv1d_80/BiasAdd/ReadVariableOp�
sequential_40/conv1d_80/BiasAddBiasAdd/sequential_40/conv1d_80/conv1d/Squeeze:output:06sequential_40/conv1d_80/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2!
sequential_40/conv1d_80/BiasAdd�
-sequential_40/max_pooling1d_80/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_40/max_pooling1d_80/ExpandDims/dim�
)sequential_40/max_pooling1d_80/ExpandDims
ExpandDims(sequential_40/conv1d_80/BiasAdd:output:06sequential_40/max_pooling1d_80/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2+
)sequential_40/max_pooling1d_80/ExpandDims�
&sequential_40/max_pooling1d_80/MaxPoolMaxPool2sequential_40/max_pooling1d_80/ExpandDims:output:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2(
&sequential_40/max_pooling1d_80/MaxPool�
&sequential_40/max_pooling1d_80/SqueezeSqueeze/sequential_40/max_pooling1d_80/MaxPool:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims
2(
&sequential_40/max_pooling1d_80/Squeeze�
$sequential_40/conv1d_81/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2&
$sequential_40/conv1d_81/Pad/paddings�
sequential_40/conv1d_81/PadPad/sequential_40/max_pooling1d_80/Squeeze:output:0-sequential_40/conv1d_81/Pad/paddings:output:0*
T0*+
_output_shapes
:���������2
sequential_40/conv1d_81/Pad�
-sequential_40/conv1d_81/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2/
-sequential_40/conv1d_81/conv1d/ExpandDims/dim�
)sequential_40/conv1d_81/conv1d/ExpandDims
ExpandDims$sequential_40/conv1d_81/Pad:output:06sequential_40/conv1d_81/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2+
)sequential_40/conv1d_81/conv1d/ExpandDims�
:sequential_40/conv1d_81/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_40_conv1d_81_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02<
:sequential_40/conv1d_81/conv1d/ExpandDims_1/ReadVariableOp�
/sequential_40/conv1d_81/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_40/conv1d_81/conv1d/ExpandDims_1/dim�
+sequential_40/conv1d_81/conv1d/ExpandDims_1
ExpandDimsBsequential_40/conv1d_81/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_40/conv1d_81/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2-
+sequential_40/conv1d_81/conv1d/ExpandDims_1�
sequential_40/conv1d_81/conv1dConv2D2sequential_40/conv1d_81/conv1d/ExpandDims:output:04sequential_40/conv1d_81/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2 
sequential_40/conv1d_81/conv1d�
&sequential_40/conv1d_81/conv1d/SqueezeSqueeze'sequential_40/conv1d_81/conv1d:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������2(
&sequential_40/conv1d_81/conv1d/Squeeze�
.sequential_40/conv1d_81/BiasAdd/ReadVariableOpReadVariableOp7sequential_40_conv1d_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_40/conv1d_81/BiasAdd/ReadVariableOp�
sequential_40/conv1d_81/BiasAddBiasAdd/sequential_40/conv1d_81/conv1d/Squeeze:output:06sequential_40/conv1d_81/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2!
sequential_40/conv1d_81/BiasAdd�
-sequential_40/max_pooling1d_81/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_40/max_pooling1d_81/ExpandDims/dim�
)sequential_40/max_pooling1d_81/ExpandDims
ExpandDims(sequential_40/conv1d_81/BiasAdd:output:06sequential_40/max_pooling1d_81/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2+
)sequential_40/max_pooling1d_81/ExpandDims�
&sequential_40/max_pooling1d_81/MaxPoolMaxPool2sequential_40/max_pooling1d_81/ExpandDims:output:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2(
&sequential_40/max_pooling1d_81/MaxPool�
&sequential_40/max_pooling1d_81/SqueezeSqueeze/sequential_40/max_pooling1d_81/MaxPool:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims
2(
&sequential_40/max_pooling1d_81/Squeeze�
sequential_40/flatten_40/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2 
sequential_40/flatten_40/Const�
 sequential_40/flatten_40/ReshapeReshape/sequential_40/max_pooling1d_81/Squeeze:output:0'sequential_40/flatten_40/Const:output:0*
T0*'
_output_shapes
:���������2"
 sequential_40/flatten_40/Reshape�
,sequential_40/dense_80/MatMul/ReadVariableOpReadVariableOp5sequential_40_dense_80_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02.
,sequential_40/dense_80/MatMul/ReadVariableOp�
sequential_40/dense_80/MatMulMatMul)sequential_40/flatten_40/Reshape:output:04sequential_40/dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
sequential_40/dense_80/MatMul�
-sequential_40/dense_80/BiasAdd/ReadVariableOpReadVariableOp6sequential_40_dense_80_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_40/dense_80/BiasAdd/ReadVariableOp�
sequential_40/dense_80/BiasAddBiasAdd'sequential_40/dense_80/MatMul:product:05sequential_40/dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2 
sequential_40/dense_80/BiasAdd�
sequential_40/dense_80/ReluRelu'sequential_40/dense_80/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
sequential_40/dense_80/Relu�
,sequential_40/dense_81/MatMul/ReadVariableOpReadVariableOp5sequential_40_dense_81_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02.
,sequential_40/dense_81/MatMul/ReadVariableOp�
sequential_40/dense_81/MatMulMatMul)sequential_40/dense_80/Relu:activations:04sequential_40/dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_40/dense_81/MatMul�
-sequential_40/dense_81/BiasAdd/ReadVariableOpReadVariableOp6sequential_40_dense_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_40/dense_81/BiasAdd/ReadVariableOp�
sequential_40/dense_81/BiasAddBiasAdd'sequential_40/dense_81/MatMul:product:05sequential_40/dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
sequential_40/dense_81/BiasAdd�
sequential_40/dense_81/EluElu'sequential_40/dense_81/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential_40/dense_81/Elu|
IdentityIdentity(sequential_40/dense_81/Elu:activations:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������:::::::::\ X
+
_output_shapes
:���������
)
_user_specified_nameconv1d_80_input
�
i
M__inference_max_pooling1d_81_layer_call_and_return_conditional_losses_5984548

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+���������������������������2

ExpandDims�
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_5984850
conv1d_80_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv1d_80_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_59845242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_nameconv1d_80_input
�
i
M__inference_max_pooling1d_80_layer_call_and_return_conditional_losses_5984533

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+���������������������������2

ExpandDims�
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
E__inference_dense_80_layer_call_and_return_conditional_losses_5984651

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_sequential_40_layer_call_fn_5984819
conv1d_80_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv1d_80_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_40_layer_call_and_return_conditional_losses_59848002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_nameconv1d_80_input
�
c
G__inference_flatten_40_layer_call_and_return_conditional_losses_5984632

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_sequential_40_layer_call_fn_5984979

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_40_layer_call_and_return_conditional_losses_59847522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_conv1d_81_layer_call_and_return_conditional_losses_5985043

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:���������2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_conv1d_81_layer_call_fn_5985052

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv1d_81_layer_call_and_return_conditional_losses_59846092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_81_layer_call_and_return_conditional_losses_5985094

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:::O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

*__inference_dense_81_layer_call_fn_5985103

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_81_layer_call_and_return_conditional_losses_59846782
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
H
,__inference_flatten_40_layer_call_fn_5985063

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_40_layer_call_and_return_conditional_losses_59846322
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_conv1d_81_layer_call_and_return_conditional_losses_5984609

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:���������2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
N
2__inference_max_pooling1d_80_layer_call_fn_5984539

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling1d_80_layer_call_and_return_conditional_losses_59845332
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
E__inference_dense_81_layer_call_and_return_conditional_losses_5984678

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:::O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
� 
�
J__inference_sequential_40_layer_call_and_return_conditional_losses_5984752

inputs
conv1d_80_5984728
conv1d_80_5984730
conv1d_81_5984734
conv1d_81_5984736
dense_80_5984741
dense_80_5984743
dense_81_5984746
dense_81_5984748
identity��!conv1d_80/StatefulPartitionedCall�!conv1d_81/StatefulPartitionedCall� dense_80/StatefulPartitionedCall� dense_81/StatefulPartitionedCall�
!conv1d_80/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_80_5984728conv1d_80_5984730*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv1d_80_layer_call_and_return_conditional_losses_59845752#
!conv1d_80/StatefulPartitionedCall�
 max_pooling1d_80/PartitionedCallPartitionedCall*conv1d_80/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling1d_80_layer_call_and_return_conditional_losses_59845332"
 max_pooling1d_80/PartitionedCall�
!conv1d_81/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_80/PartitionedCall:output:0conv1d_81_5984734conv1d_81_5984736*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv1d_81_layer_call_and_return_conditional_losses_59846092#
!conv1d_81/StatefulPartitionedCall�
 max_pooling1d_81/PartitionedCallPartitionedCall*conv1d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling1d_81_layer_call_and_return_conditional_losses_59845482"
 max_pooling1d_81/PartitionedCall�
flatten_40/PartitionedCallPartitionedCall)max_pooling1d_81/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_40_layer_call_and_return_conditional_losses_59846322
flatten_40/PartitionedCall�
 dense_80/StatefulPartitionedCallStatefulPartitionedCall#flatten_40/PartitionedCall:output:0dense_80_5984741dense_80_5984743*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_80_layer_call_and_return_conditional_losses_59846512"
 dense_80/StatefulPartitionedCall�
 dense_81/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0dense_81_5984746dense_81_5984748*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_81_layer_call_and_return_conditional_losses_59846782"
 dense_81/StatefulPartitionedCall�
IdentityIdentity)dense_81/StatefulPartitionedCall:output:0"^conv1d_80/StatefulPartitionedCall"^conv1d_81/StatefulPartitionedCall!^dense_80/StatefulPartitionedCall!^dense_81/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������::::::::2F
!conv1d_80/StatefulPartitionedCall!conv1d_80/StatefulPartitionedCall2F
!conv1d_81/StatefulPartitionedCall!conv1d_81/StatefulPartitionedCall2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�
J__inference_sequential_40_layer_call_and_return_conditional_losses_5984904

inputs9
5conv1d_80_conv1d_expanddims_1_readvariableop_resource-
)conv1d_80_biasadd_readvariableop_resource9
5conv1d_81_conv1d_expanddims_1_readvariableop_resource-
)conv1d_81_biasadd_readvariableop_resource+
'dense_80_matmul_readvariableop_resource,
(dense_80_biasadd_readvariableop_resource+
'dense_81_matmul_readvariableop_resource,
(dense_81_biasadd_readvariableop_resource
identity��
conv1d_80/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        	               2
conv1d_80/Pad/paddings�
conv1d_80/PadPadinputsconv1d_80/Pad/paddings:output:0*
T0*+
_output_shapes
:���������2
conv1d_80/Pad�
conv1d_80/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
conv1d_80/conv1d/ExpandDims/dim�
conv1d_80/conv1d/ExpandDims
ExpandDimsconv1d_80/Pad:output:0(conv1d_80/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d_80/conv1d/ExpandDims�
,conv1d_80/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_80_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02.
,conv1d_80/conv1d/ExpandDims_1/ReadVariableOp�
!conv1d_80/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_80/conv1d/ExpandDims_1/dim�
conv1d_80/conv1d/ExpandDims_1
ExpandDims4conv1d_80/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_80/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d_80/conv1d/ExpandDims_1�
conv1d_80/conv1dConv2D$conv1d_80/conv1d/ExpandDims:output:0&conv1d_80/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv1d_80/conv1d�
conv1d_80/conv1d/SqueezeSqueezeconv1d_80/conv1d:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������2
conv1d_80/conv1d/Squeeze�
 conv1d_80/BiasAdd/ReadVariableOpReadVariableOp)conv1d_80_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_80/BiasAdd/ReadVariableOp�
conv1d_80/BiasAddBiasAdd!conv1d_80/conv1d/Squeeze:output:0(conv1d_80/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2
conv1d_80/BiasAdd�
max_pooling1d_80/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_80/ExpandDims/dim�
max_pooling1d_80/ExpandDims
ExpandDimsconv1d_80/BiasAdd:output:0(max_pooling1d_80/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
max_pooling1d_80/ExpandDims�
max_pooling1d_80/MaxPoolMaxPool$max_pooling1d_80/ExpandDims:output:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
max_pooling1d_80/MaxPool�
max_pooling1d_80/SqueezeSqueeze!max_pooling1d_80/MaxPool:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims
2
max_pooling1d_80/Squeeze�
conv1d_81/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_81/Pad/paddings�
conv1d_81/PadPad!max_pooling1d_80/Squeeze:output:0conv1d_81/Pad/paddings:output:0*
T0*+
_output_shapes
:���������2
conv1d_81/Pad�
conv1d_81/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
conv1d_81/conv1d/ExpandDims/dim�
conv1d_81/conv1d/ExpandDims
ExpandDimsconv1d_81/Pad:output:0(conv1d_81/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d_81/conv1d/ExpandDims�
,conv1d_81/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_81_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_81/conv1d/ExpandDims_1/ReadVariableOp�
!conv1d_81/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_81/conv1d/ExpandDims_1/dim�
conv1d_81/conv1d/ExpandDims_1
ExpandDims4conv1d_81/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_81/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_81/conv1d/ExpandDims_1�
conv1d_81/conv1dConv2D$conv1d_81/conv1d/ExpandDims:output:0&conv1d_81/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv1d_81/conv1d�
conv1d_81/conv1d/SqueezeSqueezeconv1d_81/conv1d:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������2
conv1d_81/conv1d/Squeeze�
 conv1d_81/BiasAdd/ReadVariableOpReadVariableOp)conv1d_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_81/BiasAdd/ReadVariableOp�
conv1d_81/BiasAddBiasAdd!conv1d_81/conv1d/Squeeze:output:0(conv1d_81/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2
conv1d_81/BiasAdd�
max_pooling1d_81/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_81/ExpandDims/dim�
max_pooling1d_81/ExpandDims
ExpandDimsconv1d_81/BiasAdd:output:0(max_pooling1d_81/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
max_pooling1d_81/ExpandDims�
max_pooling1d_81/MaxPoolMaxPool$max_pooling1d_81/ExpandDims:output:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
max_pooling1d_81/MaxPool�
max_pooling1d_81/SqueezeSqueeze!max_pooling1d_81/MaxPool:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims
2
max_pooling1d_81/Squeezeu
flatten_40/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_40/Const�
flatten_40/ReshapeReshape!max_pooling1d_81/Squeeze:output:0flatten_40/Const:output:0*
T0*'
_output_shapes
:���������2
flatten_40/Reshape�
dense_80/MatMul/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_80/MatMul/ReadVariableOp�
dense_80/MatMulMatMulflatten_40/Reshape:output:0&dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_80/MatMul�
dense_80/BiasAdd/ReadVariableOpReadVariableOp(dense_80_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_80/BiasAdd/ReadVariableOp�
dense_80/BiasAddBiasAdddense_80/MatMul:product:0'dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_80/BiasAdds
dense_80/ReluReludense_80/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_80/Relu�
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_81/MatMul/ReadVariableOp�
dense_81/MatMulMatMuldense_80/Relu:activations:0&dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_81/MatMul�
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_81/BiasAdd/ReadVariableOp�
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_81/BiasAddp
dense_81/EluEludense_81/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_81/Elun
IdentityIdentitydense_81/Elu:activations:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:���������:::::::::S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

*__inference_dense_80_layer_call_fn_5985083

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_80_layer_call_and_return_conditional_losses_59846512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
N
2__inference_max_pooling1d_81_layer_call_fn_5984554

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_max_pooling1d_81_layer_call_and_return_conditional_losses_59845482
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
O
conv1d_80_input<
!serving_default_conv1d_80_input:0���������<
dense_810
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�<
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
	trainable_variables

	variables
regularization_losses
	keras_api

signatures
*{&call_and_return_all_conditional_losses
|__call__
}_default_save_signature"�8
_tf_keras_sequential�8{"class_name": "Sequential", "name": "sequential_40", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_40", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_80_input"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_80", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [10]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_80", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_81", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_81", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_40", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_80", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_81", "trainable": true, "dtype": "float32", "units": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_40", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_80_input"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_80", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [10]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_80", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_81", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_81", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_40", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_80", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_81", "trainable": true, "dtype": "float32", "units": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["mean_squared_error"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-08}}}}
�


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*~&call_and_return_all_conditional_losses
__call__"�	
_tf_keras_layer�	{"class_name": "Conv1D", "name": "conv1d_80", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_80", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [10]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 1]}}
�
trainable_variables
	variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling1D", "name": "max_pooling1d_80", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_80", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1D", "name": "conv1d_81", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_81", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 16]}}
�
trainable_variables
	variables
 regularization_losses
!	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling1D", "name": "max_pooling1d_81", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_81", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
"trainable_variables
#	variables
$regularization_losses
%	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_40", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

&kernel
'bias
(trainable_variables
)	variables
*regularization_losses
+	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_80", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_80", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
�

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_81", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_81", "trainable": true, "dtype": "float32", "units": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
�
2iter

3beta_1

4beta_2
	5decay
6learning_rate
7momentum_cachemkmlmmmn&mo'mp,mq-mrvsvtvuvv&vw'vx,vy-vz"
	optimizer
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
8non_trainable_variables
9layer_metrics
	trainable_variables

	variables

:layers
;metrics
regularization_losses
<layer_regularization_losses
|__call__
}_default_save_signature
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
&:$
2conv1d_80/kernel
:2conv1d_80/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
=layer_metrics
>non_trainable_variables
trainable_variables
	variables

?layers
@metrics
regularization_losses
Alayer_regularization_losses
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Blayer_metrics
Cnon_trainable_variables
trainable_variables
	variables

Dlayers
Emetrics
regularization_losses
Flayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$2conv1d_81/kernel
:2conv1d_81/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Glayer_metrics
Hnon_trainable_variables
trainable_variables
	variables

Ilayers
Jmetrics
regularization_losses
Klayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Llayer_metrics
Mnon_trainable_variables
trainable_variables
	variables

Nlayers
Ometrics
 regularization_losses
Player_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Qlayer_metrics
Rnon_trainable_variables
"trainable_variables
#	variables

Slayers
Tmetrics
$regularization_losses
Ulayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_80/kernel
:
2dense_80/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Vlayer_metrics
Wnon_trainable_variables
(trainable_variables
)	variables

Xlayers
Ymetrics
*regularization_losses
Zlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_81/kernel
:2dense_81/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
[layer_metrics
\non_trainable_variables
.trainable_variables
/	variables

]layers
^metrics
0regularization_losses
_layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	btotal
	ccount
d	variables
e	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	ftotal
	gcount
h
_fn_kwargs
i	variables
j	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}
:  (2total
:  (2count
.
b0
c1"
trackable_list_wrapper
-
d	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
f0
g1"
trackable_list_wrapper
-
i	variables"
_generic_user_object
,:*
2Nadam/conv1d_80/kernel/m
": 2Nadam/conv1d_80/bias/m
,:*2Nadam/conv1d_81/kernel/m
": 2Nadam/conv1d_81/bias/m
':%
2Nadam/dense_80/kernel/m
!:
2Nadam/dense_80/bias/m
':%
2Nadam/dense_81/kernel/m
!:2Nadam/dense_81/bias/m
,:*
2Nadam/conv1d_80/kernel/v
": 2Nadam/conv1d_80/bias/v
,:*2Nadam/conv1d_81/kernel/v
": 2Nadam/conv1d_81/bias/v
':%
2Nadam/dense_80/kernel/v
!:
2Nadam/dense_80/bias/v
':%
2Nadam/dense_81/kernel/v
!:2Nadam/dense_81/bias/v
�2�
J__inference_sequential_40_layer_call_and_return_conditional_losses_5984722
J__inference_sequential_40_layer_call_and_return_conditional_losses_5984695
J__inference_sequential_40_layer_call_and_return_conditional_losses_5984904
J__inference_sequential_40_layer_call_and_return_conditional_losses_5984958�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
/__inference_sequential_40_layer_call_fn_5984771
/__inference_sequential_40_layer_call_fn_5984979
/__inference_sequential_40_layer_call_fn_5984819
/__inference_sequential_40_layer_call_fn_5985000�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_5984524�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *2�/
-�*
conv1d_80_input���������
�2�
F__inference_conv1d_80_layer_call_and_return_conditional_losses_5985017�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv1d_80_layer_call_fn_5985026�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_max_pooling1d_80_layer_call_and_return_conditional_losses_5984533�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
2__inference_max_pooling1d_80_layer_call_fn_5984539�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
F__inference_conv1d_81_layer_call_and_return_conditional_losses_5985043�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv1d_81_layer_call_fn_5985052�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_max_pooling1d_81_layer_call_and_return_conditional_losses_5984548�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
2__inference_max_pooling1d_81_layer_call_fn_5984554�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
G__inference_flatten_40_layer_call_and_return_conditional_losses_5985058�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_flatten_40_layer_call_fn_5985063�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_80_layer_call_and_return_conditional_losses_5985074�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_80_layer_call_fn_5985083�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_81_layer_call_and_return_conditional_losses_5985094�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_81_layer_call_fn_5985103�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
<B:
%__inference_signature_wrapper_5984850conv1d_80_input�
"__inference__wrapped_model_5984524}&',-<�9
2�/
-�*
conv1d_80_input���������
� "3�0
.
dense_81"�
dense_81����������
F__inference_conv1d_80_layer_call_and_return_conditional_losses_5985017d3�0
)�&
$�!
inputs���������
� ")�&
�
0���������
� �
+__inference_conv1d_80_layer_call_fn_5985026W3�0
)�&
$�!
inputs���������
� "�����������
F__inference_conv1d_81_layer_call_and_return_conditional_losses_5985043d3�0
)�&
$�!
inputs���������
� ")�&
�
0���������
� �
+__inference_conv1d_81_layer_call_fn_5985052W3�0
)�&
$�!
inputs���������
� "�����������
E__inference_dense_80_layer_call_and_return_conditional_losses_5985074\&'/�,
%�"
 �
inputs���������
� "%�"
�
0���������

� }
*__inference_dense_80_layer_call_fn_5985083O&'/�,
%�"
 �
inputs���������
� "����������
�
E__inference_dense_81_layer_call_and_return_conditional_losses_5985094\,-/�,
%�"
 �
inputs���������

� "%�"
�
0���������
� }
*__inference_dense_81_layer_call_fn_5985103O,-/�,
%�"
 �
inputs���������

� "�����������
G__inference_flatten_40_layer_call_and_return_conditional_losses_5985058\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������
� 
,__inference_flatten_40_layer_call_fn_5985063O3�0
)�&
$�!
inputs���������
� "�����������
M__inference_max_pooling1d_80_layer_call_and_return_conditional_losses_5984533�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
2__inference_max_pooling1d_80_layer_call_fn_5984539wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
M__inference_max_pooling1d_81_layer_call_and_return_conditional_losses_5984548�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
2__inference_max_pooling1d_81_layer_call_fn_5984554wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
J__inference_sequential_40_layer_call_and_return_conditional_losses_5984695w&',-D�A
:�7
-�*
conv1d_80_input���������
p

 
� "%�"
�
0���������
� �
J__inference_sequential_40_layer_call_and_return_conditional_losses_5984722w&',-D�A
:�7
-�*
conv1d_80_input���������
p 

 
� "%�"
�
0���������
� �
J__inference_sequential_40_layer_call_and_return_conditional_losses_5984904n&',-;�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
J__inference_sequential_40_layer_call_and_return_conditional_losses_5984958n&',-;�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
/__inference_sequential_40_layer_call_fn_5984771j&',-D�A
:�7
-�*
conv1d_80_input���������
p

 
� "�����������
/__inference_sequential_40_layer_call_fn_5984819j&',-D�A
:�7
-�*
conv1d_80_input���������
p 

 
� "�����������
/__inference_sequential_40_layer_call_fn_5984979a&',-;�8
1�.
$�!
inputs���������
p

 
� "�����������
/__inference_sequential_40_layer_call_fn_5985000a&',-;�8
1�.
$�!
inputs���������
p 

 
� "�����������
%__inference_signature_wrapper_5984850�&',-O�L
� 
E�B
@
conv1d_80_input-�*
conv1d_80_input���������"3�0
.
dense_81"�
dense_81���������