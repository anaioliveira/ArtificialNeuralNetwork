¸	
¿£
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
dtypetype
¾
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8±

conv1d_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameconv1d_54/kernel
y
$conv1d_54/kernel/Read/ReadVariableOpReadVariableOpconv1d_54/kernel*"
_output_shapes
:
*
dtype0
t
conv1d_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_54/bias
m
"conv1d_54/bias/Read/ReadVariableOpReadVariableOpconv1d_54/bias*
_output_shapes
:*
dtype0

conv1d_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1d_55/kernel
y
$conv1d_55/kernel/Read/ReadVariableOpReadVariableOpconv1d_55/kernel*"
_output_shapes
:*
dtype0
t
conv1d_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_55/bias
m
"conv1d_55/bias/Read/ReadVariableOpReadVariableOpconv1d_55/bias*
_output_shapes
:*
dtype0
z
dense_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_54/kernel
s
#dense_54/kernel/Read/ReadVariableOpReadVariableOpdense_54/kernel*
_output_shapes

:
*
dtype0
r
dense_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_54/bias
k
!dense_54/bias/Read/ReadVariableOpReadVariableOpdense_54/bias*
_output_shapes
:
*
dtype0
z
dense_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_55/kernel
s
#dense_55/kernel/Read/ReadVariableOpReadVariableOpdense_55/kernel*
_output_shapes

:
*
dtype0
r
dense_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_55/bias
k
!dense_55/bias/Read/ReadVariableOpReadVariableOpdense_55/bias*
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

Nadam/conv1d_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameNadam/conv1d_54/kernel/m

,Nadam/conv1d_54/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_54/kernel/m*"
_output_shapes
:
*
dtype0

Nadam/conv1d_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameNadam/conv1d_54/bias/m
}
*Nadam/conv1d_54/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_54/bias/m*
_output_shapes
:*
dtype0

Nadam/conv1d_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameNadam/conv1d_55/kernel/m

,Nadam/conv1d_55/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_55/kernel/m*"
_output_shapes
:*
dtype0

Nadam/conv1d_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameNadam/conv1d_55/bias/m
}
*Nadam/conv1d_55/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv1d_55/bias/m*
_output_shapes
:*
dtype0

Nadam/dense_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_54/kernel/m

+Nadam/dense_54/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_54/kernel/m*
_output_shapes

:
*
dtype0

Nadam/dense_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_54/bias/m
{
)Nadam/dense_54/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_54/bias/m*
_output_shapes
:
*
dtype0

Nadam/dense_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_55/kernel/m

+Nadam/dense_55/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_55/kernel/m*
_output_shapes

:
*
dtype0

Nadam/dense_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_55/bias/m
{
)Nadam/dense_55/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_55/bias/m*
_output_shapes
:*
dtype0

Nadam/conv1d_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameNadam/conv1d_54/kernel/v

,Nadam/conv1d_54/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_54/kernel/v*"
_output_shapes
:
*
dtype0

Nadam/conv1d_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameNadam/conv1d_54/bias/v
}
*Nadam/conv1d_54/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_54/bias/v*
_output_shapes
:*
dtype0

Nadam/conv1d_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameNadam/conv1d_55/kernel/v

,Nadam/conv1d_55/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_55/kernel/v*"
_output_shapes
:*
dtype0

Nadam/conv1d_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameNadam/conv1d_55/bias/v
}
*Nadam/conv1d_55/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv1d_55/bias/v*
_output_shapes
:*
dtype0

Nadam/dense_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_54/kernel/v

+Nadam/dense_54/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_54/kernel/v*
_output_shapes

:
*
dtype0

Nadam/dense_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_54/bias/v
{
)Nadam/dense_54/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_54/bias/v*
_output_shapes
:
*
dtype0

Nadam/dense_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_55/kernel/v

+Nadam/dense_55/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_55/kernel/v*
_output_shapes

:
*
dtype0

Nadam/dense_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_55/bias/v
{
)Nadam/dense_55/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_55/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
6
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*É5
value¿5B¼5 Bµ5
´
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
ä
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
­
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
VARIABLE_VALUEconv1d_54/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_54/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
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
­
Blayer_metrics
Cnon_trainable_variables
trainable_variables
	variables

Dlayers
Emetrics
regularization_losses
Flayer_regularization_losses
\Z
VARIABLE_VALUEconv1d_55/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_55/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
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
­
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
­
Qlayer_metrics
Rnon_trainable_variables
"trainable_variables
#	variables

Slayers
Tmetrics
$regularization_losses
Ulayer_regularization_losses
[Y
VARIABLE_VALUEdense_54/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_54/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1

&0
'1
 
­
Vlayer_metrics
Wnon_trainable_variables
(trainable_variables
)	variables

Xlayers
Ymetrics
*regularization_losses
Zlayer_regularization_losses
[Y
VARIABLE_VALUEdense_55/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_55/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 
­
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
~
VARIABLE_VALUENadam/conv1d_54/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/conv1d_54/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUENadam/conv1d_55/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/conv1d_55/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_54/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_54/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_55/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_55/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUENadam/conv1d_54/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/conv1d_54/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUENadam/conv1d_55/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/conv1d_55/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_54/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_54/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_55/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_55/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv1d_54_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
Ï
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_54_inputconv1d_54/kernelconv1d_54/biasconv1d_55/kernelconv1d_55/biasdense_54/kerneldense_54/biasdense_55/kerneldense_55/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_4086980
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ò
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv1d_54/kernel/Read/ReadVariableOp"conv1d_54/bias/Read/ReadVariableOp$conv1d_55/kernel/Read/ReadVariableOp"conv1d_55/bias/Read/ReadVariableOp#dense_54/kernel/Read/ReadVariableOp!dense_54/bias/Read/ReadVariableOp#dense_55/kernel/Read/ReadVariableOp!dense_55/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Nadam/conv1d_54/kernel/m/Read/ReadVariableOp*Nadam/conv1d_54/bias/m/Read/ReadVariableOp,Nadam/conv1d_55/kernel/m/Read/ReadVariableOp*Nadam/conv1d_55/bias/m/Read/ReadVariableOp+Nadam/dense_54/kernel/m/Read/ReadVariableOp)Nadam/dense_54/bias/m/Read/ReadVariableOp+Nadam/dense_55/kernel/m/Read/ReadVariableOp)Nadam/dense_55/bias/m/Read/ReadVariableOp,Nadam/conv1d_54/kernel/v/Read/ReadVariableOp*Nadam/conv1d_54/bias/v/Read/ReadVariableOp,Nadam/conv1d_55/kernel/v/Read/ReadVariableOp*Nadam/conv1d_55/bias/v/Read/ReadVariableOp+Nadam/dense_54/kernel/v/Read/ReadVariableOp)Nadam/dense_54/bias/v/Read/ReadVariableOp+Nadam/dense_55/kernel/v/Read/ReadVariableOp)Nadam/dense_55/bias/v/Read/ReadVariableOpConst*/
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_4087358
Å
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_54/kernelconv1d_54/biasconv1d_55/kernelconv1d_55/biasdense_54/kerneldense_54/biasdense_55/kerneldense_55/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcounttotal_1count_1Nadam/conv1d_54/kernel/mNadam/conv1d_54/bias/mNadam/conv1d_55/kernel/mNadam/conv1d_55/bias/mNadam/dense_54/kernel/mNadam/dense_54/bias/mNadam/dense_55/kernel/mNadam/dense_55/bias/mNadam/conv1d_54/kernel/vNadam/conv1d_54/bias/vNadam/conv1d_55/kernel/vNadam/conv1d_55/bias/vNadam/dense_54/kernel/vNadam/dense_54/bias/vNadam/dense_55/kernel/vNadam/dense_55/bias/v*.
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_4087470¦ü
¦
­
E__inference_dense_55_layer_call_and_return_conditional_losses_4087224

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ª
­
E__inference_dense_54_layer_call_and_return_conditional_losses_4086781

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
­
E__inference_dense_54_layer_call_and_return_conditional_losses_4087204

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á 
´
J__inference_sequential_27_layer_call_and_return_conditional_losses_4086852
conv1d_54_input
conv1d_54_4086828
conv1d_54_4086830
conv1d_55_4086834
conv1d_55_4086836
dense_54_4086841
dense_54_4086843
dense_55_4086846
dense_55_4086848
identity¢!conv1d_54/StatefulPartitionedCall¢!conv1d_55/StatefulPartitionedCall¢ dense_54/StatefulPartitionedCall¢ dense_55/StatefulPartitionedCall©
!conv1d_54/StatefulPartitionedCallStatefulPartitionedCallconv1d_54_inputconv1d_54_4086828conv1d_54_4086830*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv1d_54_layer_call_and_return_conditional_losses_40867052#
!conv1d_54/StatefulPartitionedCall
 max_pooling1d_54/PartitionedCallPartitionedCall*conv1d_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling1d_54_layer_call_and_return_conditional_losses_40866632"
 max_pooling1d_54/PartitionedCallÃ
!conv1d_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_54/PartitionedCall:output:0conv1d_55_4086834conv1d_55_4086836*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv1d_55_layer_call_and_return_conditional_losses_40867392#
!conv1d_55/StatefulPartitionedCall
 max_pooling1d_55/PartitionedCallPartitionedCall*conv1d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling1d_55_layer_call_and_return_conditional_losses_40866782"
 max_pooling1d_55/PartitionedCallþ
flatten_27/PartitionedCallPartitionedCall)max_pooling1d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_27_layer_call_and_return_conditional_losses_40867622
flatten_27/PartitionedCall´
 dense_54/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_54_4086841dense_54_4086843*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_54_layer_call_and_return_conditional_losses_40867812"
 dense_54/StatefulPartitionedCallº
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_4086846dense_55_4086848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_55_layer_call_and_return_conditional_losses_40868082"
 dense_55/StatefulPartitionedCall
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0"^conv1d_54/StatefulPartitionedCall"^conv1d_55/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv1d_54/StatefulPartitionedCall!conv1d_54/StatefulPartitionedCall2F
!conv1d_55/StatefulPartitionedCall!conv1d_55/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameconv1d_54_input
²
»
F__inference_conv1d_55_layer_call_and_return_conditional_losses_4087173

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
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
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:::S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ

+__inference_conv1d_55_layer_call_fn_4087182

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv1d_55_layer_call_and_return_conditional_losses_40867392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
i
M__inference_max_pooling1d_54_layer_call_and_return_conditional_losses_4086663

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDims±
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦ 
«
J__inference_sequential_27_layer_call_and_return_conditional_losses_4086882

inputs
conv1d_54_4086858
conv1d_54_4086860
conv1d_55_4086864
conv1d_55_4086866
dense_54_4086871
dense_54_4086873
dense_55_4086876
dense_55_4086878
identity¢!conv1d_54/StatefulPartitionedCall¢!conv1d_55/StatefulPartitionedCall¢ dense_54/StatefulPartitionedCall¢ dense_55/StatefulPartitionedCall 
!conv1d_54/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_54_4086858conv1d_54_4086860*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv1d_54_layer_call_and_return_conditional_losses_40867052#
!conv1d_54/StatefulPartitionedCall
 max_pooling1d_54/PartitionedCallPartitionedCall*conv1d_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling1d_54_layer_call_and_return_conditional_losses_40866632"
 max_pooling1d_54/PartitionedCallÃ
!conv1d_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_54/PartitionedCall:output:0conv1d_55_4086864conv1d_55_4086866*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv1d_55_layer_call_and_return_conditional_losses_40867392#
!conv1d_55/StatefulPartitionedCall
 max_pooling1d_55/PartitionedCallPartitionedCall*conv1d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling1d_55_layer_call_and_return_conditional_losses_40866782"
 max_pooling1d_55/PartitionedCallþ
flatten_27/PartitionedCallPartitionedCall)max_pooling1d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_27_layer_call_and_return_conditional_losses_40867622
flatten_27/PartitionedCall´
 dense_54/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_54_4086871dense_54_4086873*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_54_layer_call_and_return_conditional_losses_40867812"
 dense_54/StatefulPartitionedCallº
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_4086876dense_55_4086878*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_55_layer_call_and_return_conditional_losses_40868082"
 dense_55/StatefulPartitionedCall
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0"^conv1d_54/StatefulPartitionedCall"^conv1d_55/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv1d_54/StatefulPartitionedCall!conv1d_54/StatefulPartitionedCall2F
!conv1d_55/StatefulPartitionedCall!conv1d_55/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ý
%__inference_signature_wrapper_4086980
conv1d_54_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallconv1d_54_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_40866542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameconv1d_54_input
¦
­
E__inference_dense_55_layer_call_and_return_conditional_losses_4086808

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
²
»
F__inference_conv1d_54_layer_call_and_return_conditional_losses_4086705

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        	               2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
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
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:::S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
Þ
/__inference_sequential_27_layer_call_fn_4087130

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_27_layer_call_and_return_conditional_losses_40869302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
»
F__inference_conv1d_55_layer_call_and_return_conditional_losses_4086739

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
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
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:::S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
ç
/__inference_sequential_27_layer_call_fn_4086901
conv1d_54_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallconv1d_54_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_27_layer_call_and_return_conditional_losses_40868822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameconv1d_54_input
 
H
,__inference_flatten_27_layer_call_fn_4087193

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_27_layer_call_and_return_conditional_losses_40867622
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯?
ó
J__inference_sequential_27_layer_call_and_return_conditional_losses_4087088

inputs9
5conv1d_54_conv1d_expanddims_1_readvariableop_resource-
)conv1d_54_biasadd_readvariableop_resource9
5conv1d_55_conv1d_expanddims_1_readvariableop_resource-
)conv1d_55_biasadd_readvariableop_resource+
'dense_54_matmul_readvariableop_resource,
(dense_54_biasadd_readvariableop_resource+
'dense_55_matmul_readvariableop_resource,
(dense_55_biasadd_readvariableop_resource
identity
conv1d_54/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        	               2
conv1d_54/Pad/paddings
conv1d_54/PadPadinputsconv1d_54/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_54/Pad
conv1d_54/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_54/conv1d/ExpandDims/dimÄ
conv1d_54/conv1d/ExpandDims
ExpandDimsconv1d_54/Pad:output:0(conv1d_54/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_54/conv1d/ExpandDimsÖ
,conv1d_54/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_54_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02.
,conv1d_54/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_54/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_54/conv1d/ExpandDims_1/dimß
conv1d_54/conv1d/ExpandDims_1
ExpandDims4conv1d_54/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_54/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d_54/conv1d/ExpandDims_1ß
conv1d_54/conv1dConv2D$conv1d_54/conv1d/ExpandDims:output:0&conv1d_54/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d_54/conv1d°
conv1d_54/conv1d/SqueezeSqueezeconv1d_54/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_54/conv1d/Squeezeª
 conv1d_54/BiasAdd/ReadVariableOpReadVariableOp)conv1d_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_54/BiasAdd/ReadVariableOp´
conv1d_54/BiasAddBiasAdd!conv1d_54/conv1d/Squeeze:output:0(conv1d_54/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_54/BiasAdd
max_pooling1d_54/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_54/ExpandDims/dimÈ
max_pooling1d_54/ExpandDims
ExpandDimsconv1d_54/BiasAdd:output:0(max_pooling1d_54/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
max_pooling1d_54/ExpandDimsÒ
max_pooling1d_54/MaxPoolMaxPool$max_pooling1d_54/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling1d_54/MaxPool¯
max_pooling1d_54/SqueezeSqueeze!max_pooling1d_54/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
max_pooling1d_54/Squeeze
conv1d_55/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_55/Pad/paddings
conv1d_55/PadPad!max_pooling1d_54/Squeeze:output:0conv1d_55/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_55/Pad
conv1d_55/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_55/conv1d/ExpandDims/dimÄ
conv1d_55/conv1d/ExpandDims
ExpandDimsconv1d_55/Pad:output:0(conv1d_55/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_55/conv1d/ExpandDimsÖ
,conv1d_55/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_55_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_55/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_55/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_55/conv1d/ExpandDims_1/dimß
conv1d_55/conv1d/ExpandDims_1
ExpandDims4conv1d_55/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_55/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_55/conv1d/ExpandDims_1ß
conv1d_55/conv1dConv2D$conv1d_55/conv1d/ExpandDims:output:0&conv1d_55/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d_55/conv1d°
conv1d_55/conv1d/SqueezeSqueezeconv1d_55/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_55/conv1d/Squeezeª
 conv1d_55/BiasAdd/ReadVariableOpReadVariableOp)conv1d_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_55/BiasAdd/ReadVariableOp´
conv1d_55/BiasAddBiasAdd!conv1d_55/conv1d/Squeeze:output:0(conv1d_55/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_55/BiasAdd
max_pooling1d_55/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_55/ExpandDims/dimÈ
max_pooling1d_55/ExpandDims
ExpandDimsconv1d_55/BiasAdd:output:0(max_pooling1d_55/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
max_pooling1d_55/ExpandDimsÒ
max_pooling1d_55/MaxPoolMaxPool$max_pooling1d_55/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling1d_55/MaxPool¯
max_pooling1d_55/SqueezeSqueeze!max_pooling1d_55/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
max_pooling1d_55/Squeezeu
flatten_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_27/Const£
flatten_27/ReshapeReshape!max_pooling1d_55/Squeeze:output:0flatten_27/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_27/Reshape¨
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_54/MatMul/ReadVariableOp£
dense_54/MatMulMatMulflatten_27/Reshape:output:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_54/MatMul§
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_54/BiasAdd/ReadVariableOp¥
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_54/BiasAdds
dense_54/ReluReludense_54/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_54/Relu¨
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_55/MatMul/ReadVariableOp£
dense_55/MatMulMatMuldense_54/Relu:activations:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_55/MatMul§
dense_55/BiasAdd/ReadVariableOpReadVariableOp(dense_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_55/BiasAdd/ReadVariableOp¥
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_55/BiasAddp
dense_55/EluEludense_55/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_55/Elun
IdentityIdentitydense_55/Elu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::::::S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
c
G__inference_flatten_27_layer_call_and_return_conditional_losses_4087188

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
N
2__inference_max_pooling1d_54_layer_call_fn_4086669

inputs
identityá
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling1d_54_layer_call_and_return_conditional_losses_40866632
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
J

 __inference__traced_save_4087358
file_prefix/
+savev2_conv1d_54_kernel_read_readvariableop-
)savev2_conv1d_54_bias_read_readvariableop/
+savev2_conv1d_55_kernel_read_readvariableop-
)savev2_conv1d_55_bias_read_readvariableop.
*savev2_dense_54_kernel_read_readvariableop,
(savev2_dense_54_bias_read_readvariableop.
*savev2_dense_55_kernel_read_readvariableop,
(savev2_dense_55_bias_read_readvariableop)
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
3savev2_nadam_conv1d_54_kernel_m_read_readvariableop5
1savev2_nadam_conv1d_54_bias_m_read_readvariableop7
3savev2_nadam_conv1d_55_kernel_m_read_readvariableop5
1savev2_nadam_conv1d_55_bias_m_read_readvariableop6
2savev2_nadam_dense_54_kernel_m_read_readvariableop4
0savev2_nadam_dense_54_bias_m_read_readvariableop6
2savev2_nadam_dense_55_kernel_m_read_readvariableop4
0savev2_nadam_dense_55_bias_m_read_readvariableop7
3savev2_nadam_conv1d_54_kernel_v_read_readvariableop5
1savev2_nadam_conv1d_54_bias_v_read_readvariableop7
3savev2_nadam_conv1d_55_kernel_v_read_readvariableop5
1savev2_nadam_conv1d_55_bias_v_read_readvariableop6
2savev2_nadam_dense_54_kernel_v_read_readvariableop4
0savev2_nadam_dense_54_bias_v_read_readvariableop6
2savev2_nadam_dense_55_kernel_v_read_readvariableop4
0savev2_nadam_dense_55_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b191dfb39ccc4bbab1ea10ef4ddf0b01/part2	
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameû
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*
valueB#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÎ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesþ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv1d_54_kernel_read_readvariableop)savev2_conv1d_54_bias_read_readvariableop+savev2_conv1d_55_kernel_read_readvariableop)savev2_conv1d_55_bias_read_readvariableop*savev2_dense_54_kernel_read_readvariableop(savev2_dense_54_bias_read_readvariableop*savev2_dense_55_kernel_read_readvariableop(savev2_dense_55_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_nadam_conv1d_54_kernel_m_read_readvariableop1savev2_nadam_conv1d_54_bias_m_read_readvariableop3savev2_nadam_conv1d_55_kernel_m_read_readvariableop1savev2_nadam_conv1d_55_bias_m_read_readvariableop2savev2_nadam_dense_54_kernel_m_read_readvariableop0savev2_nadam_dense_54_bias_m_read_readvariableop2savev2_nadam_dense_55_kernel_m_read_readvariableop0savev2_nadam_dense_55_bias_m_read_readvariableop3savev2_nadam_conv1d_54_kernel_v_read_readvariableop1savev2_nadam_conv1d_54_bias_v_read_readvariableop3savev2_nadam_conv1d_55_kernel_v_read_readvariableop1savev2_nadam_conv1d_55_bias_v_read_readvariableop2savev2_nadam_dense_54_kernel_v_read_readvariableop0savev2_nadam_dense_54_bias_v_read_readvariableop2savev2_nadam_dense_55_kernel_v_read_readvariableop0savev2_nadam_dense_55_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*
_input_shapesó
ð: :
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
éN
Ä
"__inference__wrapped_model_4086654
conv1d_54_inputG
Csequential_27_conv1d_54_conv1d_expanddims_1_readvariableop_resource;
7sequential_27_conv1d_54_biasadd_readvariableop_resourceG
Csequential_27_conv1d_55_conv1d_expanddims_1_readvariableop_resource;
7sequential_27_conv1d_55_biasadd_readvariableop_resource9
5sequential_27_dense_54_matmul_readvariableop_resource:
6sequential_27_dense_54_biasadd_readvariableop_resource9
5sequential_27_dense_55_matmul_readvariableop_resource:
6sequential_27_dense_55_biasadd_readvariableop_resource
identityµ
$sequential_27/conv1d_54/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        	               2&
$sequential_27/conv1d_54/Pad/paddings·
sequential_27/conv1d_54/PadPadconv1d_54_input-sequential_27/conv1d_54/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_27/conv1d_54/Pad©
-sequential_27/conv1d_54/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2/
-sequential_27/conv1d_54/conv1d/ExpandDims/dimü
)sequential_27/conv1d_54/conv1d/ExpandDims
ExpandDims$sequential_27/conv1d_54/Pad:output:06sequential_27/conv1d_54/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_27/conv1d_54/conv1d/ExpandDims
:sequential_27/conv1d_54/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_27_conv1d_54_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02<
:sequential_27/conv1d_54/conv1d/ExpandDims_1/ReadVariableOp¤
/sequential_27/conv1d_54/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_27/conv1d_54/conv1d/ExpandDims_1/dim
+sequential_27/conv1d_54/conv1d/ExpandDims_1
ExpandDimsBsequential_27/conv1d_54/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_27/conv1d_54/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2-
+sequential_27/conv1d_54/conv1d/ExpandDims_1
sequential_27/conv1d_54/conv1dConv2D2sequential_27/conv1d_54/conv1d/ExpandDims:output:04sequential_27/conv1d_54/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2 
sequential_27/conv1d_54/conv1dÚ
&sequential_27/conv1d_54/conv1d/SqueezeSqueeze'sequential_27/conv1d_54/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2(
&sequential_27/conv1d_54/conv1d/SqueezeÔ
.sequential_27/conv1d_54/BiasAdd/ReadVariableOpReadVariableOp7sequential_27_conv1d_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_27/conv1d_54/BiasAdd/ReadVariableOpì
sequential_27/conv1d_54/BiasAddBiasAdd/sequential_27/conv1d_54/conv1d/Squeeze:output:06sequential_27/conv1d_54/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_27/conv1d_54/BiasAdd 
-sequential_27/max_pooling1d_54/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_27/max_pooling1d_54/ExpandDims/dim
)sequential_27/max_pooling1d_54/ExpandDims
ExpandDims(sequential_27/conv1d_54/BiasAdd:output:06sequential_27/max_pooling1d_54/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_27/max_pooling1d_54/ExpandDimsü
&sequential_27/max_pooling1d_54/MaxPoolMaxPool2sequential_27/max_pooling1d_54/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2(
&sequential_27/max_pooling1d_54/MaxPoolÙ
&sequential_27/max_pooling1d_54/SqueezeSqueeze/sequential_27/max_pooling1d_54/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2(
&sequential_27/max_pooling1d_54/Squeezeµ
$sequential_27/conv1d_55/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2&
$sequential_27/conv1d_55/Pad/paddings×
sequential_27/conv1d_55/PadPad/sequential_27/max_pooling1d_54/Squeeze:output:0-sequential_27/conv1d_55/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_27/conv1d_55/Pad©
-sequential_27/conv1d_55/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2/
-sequential_27/conv1d_55/conv1d/ExpandDims/dimü
)sequential_27/conv1d_55/conv1d/ExpandDims
ExpandDims$sequential_27/conv1d_55/Pad:output:06sequential_27/conv1d_55/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_27/conv1d_55/conv1d/ExpandDims
:sequential_27/conv1d_55/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpCsequential_27_conv1d_55_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02<
:sequential_27/conv1d_55/conv1d/ExpandDims_1/ReadVariableOp¤
/sequential_27/conv1d_55/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_27/conv1d_55/conv1d/ExpandDims_1/dim
+sequential_27/conv1d_55/conv1d/ExpandDims_1
ExpandDimsBsequential_27/conv1d_55/conv1d/ExpandDims_1/ReadVariableOp:value:08sequential_27/conv1d_55/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2-
+sequential_27/conv1d_55/conv1d/ExpandDims_1
sequential_27/conv1d_55/conv1dConv2D2sequential_27/conv1d_55/conv1d/ExpandDims:output:04sequential_27/conv1d_55/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2 
sequential_27/conv1d_55/conv1dÚ
&sequential_27/conv1d_55/conv1d/SqueezeSqueeze'sequential_27/conv1d_55/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2(
&sequential_27/conv1d_55/conv1d/SqueezeÔ
.sequential_27/conv1d_55/BiasAdd/ReadVariableOpReadVariableOp7sequential_27_conv1d_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_27/conv1d_55/BiasAdd/ReadVariableOpì
sequential_27/conv1d_55/BiasAddBiasAdd/sequential_27/conv1d_55/conv1d/Squeeze:output:06sequential_27/conv1d_55/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_27/conv1d_55/BiasAdd 
-sequential_27/max_pooling1d_55/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_27/max_pooling1d_55/ExpandDims/dim
)sequential_27/max_pooling1d_55/ExpandDims
ExpandDims(sequential_27/conv1d_55/BiasAdd:output:06sequential_27/max_pooling1d_55/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_27/max_pooling1d_55/ExpandDimsü
&sequential_27/max_pooling1d_55/MaxPoolMaxPool2sequential_27/max_pooling1d_55/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2(
&sequential_27/max_pooling1d_55/MaxPoolÙ
&sequential_27/max_pooling1d_55/SqueezeSqueeze/sequential_27/max_pooling1d_55/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2(
&sequential_27/max_pooling1d_55/Squeeze
sequential_27/flatten_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2 
sequential_27/flatten_27/ConstÛ
 sequential_27/flatten_27/ReshapeReshape/sequential_27/max_pooling1d_55/Squeeze:output:0'sequential_27/flatten_27/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_27/flatten_27/ReshapeÒ
,sequential_27/dense_54/MatMul/ReadVariableOpReadVariableOp5sequential_27_dense_54_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02.
,sequential_27/dense_54/MatMul/ReadVariableOpÛ
sequential_27/dense_54/MatMulMatMul)sequential_27/flatten_27/Reshape:output:04sequential_27/dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_27/dense_54/MatMulÑ
-sequential_27/dense_54/BiasAdd/ReadVariableOpReadVariableOp6sequential_27_dense_54_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_27/dense_54/BiasAdd/ReadVariableOpÝ
sequential_27/dense_54/BiasAddBiasAdd'sequential_27/dense_54/MatMul:product:05sequential_27/dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2 
sequential_27/dense_54/BiasAdd
sequential_27/dense_54/ReluRelu'sequential_27/dense_54/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_27/dense_54/ReluÒ
,sequential_27/dense_55/MatMul/ReadVariableOpReadVariableOp5sequential_27_dense_55_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02.
,sequential_27/dense_55/MatMul/ReadVariableOpÛ
sequential_27/dense_55/MatMulMatMul)sequential_27/dense_54/Relu:activations:04sequential_27/dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_27/dense_55/MatMulÑ
-sequential_27/dense_55/BiasAdd/ReadVariableOpReadVariableOp6sequential_27_dense_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_27/dense_55/BiasAdd/ReadVariableOpÝ
sequential_27/dense_55/BiasAddBiasAdd'sequential_27/dense_55/MatMul:product:05sequential_27/dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_27/dense_55/BiasAdd
sequential_27/dense_55/EluElu'sequential_27/dense_55/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_27/dense_55/Elu|
IdentityIdentity(sequential_27/dense_55/Elu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::::::\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameconv1d_54_input
ý
N
2__inference_max_pooling1d_55_layer_call_fn_4086684

inputs
identityá
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling1d_55_layer_call_and_return_conditional_losses_40866782
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
ç
/__inference_sequential_27_layer_call_fn_4086949
conv1d_54_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallconv1d_54_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_27_layer_call_and_return_conditional_losses_40869302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameconv1d_54_input
ñ

+__inference_conv1d_54_layer_call_fn_4087156

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv1d_54_layer_call_and_return_conditional_losses_40867052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á 
´
J__inference_sequential_27_layer_call_and_return_conditional_losses_4086825
conv1d_54_input
conv1d_54_4086716
conv1d_54_4086718
conv1d_55_4086750
conv1d_55_4086752
dense_54_4086792
dense_54_4086794
dense_55_4086819
dense_55_4086821
identity¢!conv1d_54/StatefulPartitionedCall¢!conv1d_55/StatefulPartitionedCall¢ dense_54/StatefulPartitionedCall¢ dense_55/StatefulPartitionedCall©
!conv1d_54/StatefulPartitionedCallStatefulPartitionedCallconv1d_54_inputconv1d_54_4086716conv1d_54_4086718*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv1d_54_layer_call_and_return_conditional_losses_40867052#
!conv1d_54/StatefulPartitionedCall
 max_pooling1d_54/PartitionedCallPartitionedCall*conv1d_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling1d_54_layer_call_and_return_conditional_losses_40866632"
 max_pooling1d_54/PartitionedCallÃ
!conv1d_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_54/PartitionedCall:output:0conv1d_55_4086750conv1d_55_4086752*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv1d_55_layer_call_and_return_conditional_losses_40867392#
!conv1d_55/StatefulPartitionedCall
 max_pooling1d_55/PartitionedCallPartitionedCall*conv1d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling1d_55_layer_call_and_return_conditional_losses_40866782"
 max_pooling1d_55/PartitionedCallþ
flatten_27/PartitionedCallPartitionedCall)max_pooling1d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_27_layer_call_and_return_conditional_losses_40867622
flatten_27/PartitionedCall´
 dense_54/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_54_4086792dense_54_4086794*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_54_layer_call_and_return_conditional_losses_40867812"
 dense_54/StatefulPartitionedCallº
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_4086819dense_55_4086821*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_55_layer_call_and_return_conditional_losses_40868082"
 dense_55/StatefulPartitionedCall
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0"^conv1d_54/StatefulPartitionedCall"^conv1d_55/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv1d_54/StatefulPartitionedCall!conv1d_54/StatefulPartitionedCall2F
!conv1d_55/StatefulPartitionedCall!conv1d_55/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameconv1d_54_input
Þ

*__inference_dense_54_layer_call_fn_4087213

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_54_layer_call_and_return_conditional_losses_40867812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ

*__inference_dense_55_layer_call_fn_4087233

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_55_layer_call_and_return_conditional_losses_40868082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ê
i
M__inference_max_pooling1d_55_layer_call_and_return_conditional_losses_4086678

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDims±
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯?
ó
J__inference_sequential_27_layer_call_and_return_conditional_losses_4087034

inputs9
5conv1d_54_conv1d_expanddims_1_readvariableop_resource-
)conv1d_54_biasadd_readvariableop_resource9
5conv1d_55_conv1d_expanddims_1_readvariableop_resource-
)conv1d_55_biasadd_readvariableop_resource+
'dense_54_matmul_readvariableop_resource,
(dense_54_biasadd_readvariableop_resource+
'dense_55_matmul_readvariableop_resource,
(dense_55_biasadd_readvariableop_resource
identity
conv1d_54/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        	               2
conv1d_54/Pad/paddings
conv1d_54/PadPadinputsconv1d_54/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_54/Pad
conv1d_54/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_54/conv1d/ExpandDims/dimÄ
conv1d_54/conv1d/ExpandDims
ExpandDimsconv1d_54/Pad:output:0(conv1d_54/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_54/conv1d/ExpandDimsÖ
,conv1d_54/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_54_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02.
,conv1d_54/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_54/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_54/conv1d/ExpandDims_1/dimß
conv1d_54/conv1d/ExpandDims_1
ExpandDims4conv1d_54/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_54/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d_54/conv1d/ExpandDims_1ß
conv1d_54/conv1dConv2D$conv1d_54/conv1d/ExpandDims:output:0&conv1d_54/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d_54/conv1d°
conv1d_54/conv1d/SqueezeSqueezeconv1d_54/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_54/conv1d/Squeezeª
 conv1d_54/BiasAdd/ReadVariableOpReadVariableOp)conv1d_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_54/BiasAdd/ReadVariableOp´
conv1d_54/BiasAddBiasAdd!conv1d_54/conv1d/Squeeze:output:0(conv1d_54/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_54/BiasAdd
max_pooling1d_54/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_54/ExpandDims/dimÈ
max_pooling1d_54/ExpandDims
ExpandDimsconv1d_54/BiasAdd:output:0(max_pooling1d_54/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
max_pooling1d_54/ExpandDimsÒ
max_pooling1d_54/MaxPoolMaxPool$max_pooling1d_54/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling1d_54/MaxPool¯
max_pooling1d_54/SqueezeSqueeze!max_pooling1d_54/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
max_pooling1d_54/Squeeze
conv1d_55/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       2
conv1d_55/Pad/paddings
conv1d_55/PadPad!max_pooling1d_54/Squeeze:output:0conv1d_55/Pad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_55/Pad
conv1d_55/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2!
conv1d_55/conv1d/ExpandDims/dimÄ
conv1d_55/conv1d/ExpandDims
ExpandDimsconv1d_55/Pad:output:0(conv1d_55/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_55/conv1d/ExpandDimsÖ
,conv1d_55/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_55_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_55/conv1d/ExpandDims_1/ReadVariableOp
!conv1d_55/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_55/conv1d/ExpandDims_1/dimß
conv1d_55/conv1d/ExpandDims_1
ExpandDims4conv1d_55/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_55/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_55/conv1d/ExpandDims_1ß
conv1d_55/conv1dConv2D$conv1d_55/conv1d/ExpandDims:output:0&conv1d_55/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d_55/conv1d°
conv1d_55/conv1d/SqueezeSqueezeconv1d_55/conv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_55/conv1d/Squeezeª
 conv1d_55/BiasAdd/ReadVariableOpReadVariableOp)conv1d_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1d_55/BiasAdd/ReadVariableOp´
conv1d_55/BiasAddBiasAdd!conv1d_55/conv1d/Squeeze:output:0(conv1d_55/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d_55/BiasAdd
max_pooling1d_55/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_55/ExpandDims/dimÈ
max_pooling1d_55/ExpandDims
ExpandDimsconv1d_55/BiasAdd:output:0(max_pooling1d_55/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
max_pooling1d_55/ExpandDimsÒ
max_pooling1d_55/MaxPoolMaxPool$max_pooling1d_55/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling1d_55/MaxPool¯
max_pooling1d_55/SqueezeSqueeze!max_pooling1d_55/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
max_pooling1d_55/Squeezeu
flatten_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_27/Const£
flatten_27/ReshapeReshape!max_pooling1d_55/Squeeze:output:0flatten_27/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_27/Reshape¨
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_54/MatMul/ReadVariableOp£
dense_54/MatMulMatMulflatten_27/Reshape:output:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_54/MatMul§
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_54/BiasAdd/ReadVariableOp¥
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_54/BiasAdds
dense_54/ReluReludense_54/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_54/Relu¨
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_55/MatMul/ReadVariableOp£
dense_55/MatMulMatMuldense_54/Relu:activations:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_55/MatMul§
dense_55/BiasAdd/ReadVariableOpReadVariableOp(dense_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_55/BiasAdd/ReadVariableOp¥
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_55/BiasAddp
dense_55/EluEludense_55/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_55/Elun
IdentityIdentitydense_55/Elu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::::::S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
c
G__inference_flatten_27_layer_call_and_return_conditional_losses_4086762

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
Þ
/__inference_sequential_27_layer_call_fn_4087109

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_27_layer_call_and_return_conditional_losses_40868822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
Ú
#__inference__traced_restore_4087470
file_prefix%
!assignvariableop_conv1d_54_kernel%
!assignvariableop_1_conv1d_54_bias'
#assignvariableop_2_conv1d_55_kernel%
!assignvariableop_3_conv1d_55_bias&
"assignvariableop_4_dense_54_kernel$
 assignvariableop_5_dense_54_bias&
"assignvariableop_6_dense_55_kernel$
 assignvariableop_7_dense_55_bias!
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
,assignvariableop_18_nadam_conv1d_54_kernel_m.
*assignvariableop_19_nadam_conv1d_54_bias_m0
,assignvariableop_20_nadam_conv1d_55_kernel_m.
*assignvariableop_21_nadam_conv1d_55_bias_m/
+assignvariableop_22_nadam_dense_54_kernel_m-
)assignvariableop_23_nadam_dense_54_bias_m/
+assignvariableop_24_nadam_dense_55_kernel_m-
)assignvariableop_25_nadam_dense_55_bias_m0
,assignvariableop_26_nadam_conv1d_54_kernel_v.
*assignvariableop_27_nadam_conv1d_54_bias_v0
,assignvariableop_28_nadam_conv1d_55_kernel_v.
*assignvariableop_29_nadam_conv1d_55_bias_v/
+assignvariableop_30_nadam_dense_54_kernel_v-
)assignvariableop_31_nadam_dense_54_bias_v/
+assignvariableop_32_nadam_dense_55_kernel_v-
)assignvariableop_33_nadam_dense_55_bias_v
identity_35¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*
valueB#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÔ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÝ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¢
_output_shapes
:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv1d_54_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv1d_54_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv1d_55_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv1d_55_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_54_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_54_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_55_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_55_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8¢
AssignVariableOp_8AssignVariableOpassignvariableop_8_nadam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¤
AssignVariableOp_9AssignVariableOpassignvariableop_9_nadam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¨
AssignVariableOp_10AssignVariableOp assignvariableop_10_nadam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11§
AssignVariableOp_11AssignVariableOpassignvariableop_11_nadam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¯
AssignVariableOp_12AssignVariableOp'assignvariableop_12_nadam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13°
AssignVariableOp_13AssignVariableOp(assignvariableop_13_nadam_momentum_cacheIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¡
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¡
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16£
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17£
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18´
AssignVariableOp_18AssignVariableOp,assignvariableop_18_nadam_conv1d_54_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19²
AssignVariableOp_19AssignVariableOp*assignvariableop_19_nadam_conv1d_54_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20´
AssignVariableOp_20AssignVariableOp,assignvariableop_20_nadam_conv1d_55_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21²
AssignVariableOp_21AssignVariableOp*assignvariableop_21_nadam_conv1d_55_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22³
AssignVariableOp_22AssignVariableOp+assignvariableop_22_nadam_dense_54_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23±
AssignVariableOp_23AssignVariableOp)assignvariableop_23_nadam_dense_54_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24³
AssignVariableOp_24AssignVariableOp+assignvariableop_24_nadam_dense_55_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25±
AssignVariableOp_25AssignVariableOp)assignvariableop_25_nadam_dense_55_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26´
AssignVariableOp_26AssignVariableOp,assignvariableop_26_nadam_conv1d_54_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27²
AssignVariableOp_27AssignVariableOp*assignvariableop_27_nadam_conv1d_54_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28´
AssignVariableOp_28AssignVariableOp,assignvariableop_28_nadam_conv1d_55_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29²
AssignVariableOp_29AssignVariableOp*assignvariableop_29_nadam_conv1d_55_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30³
AssignVariableOp_30AssignVariableOp+assignvariableop_30_nadam_dense_54_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31±
AssignVariableOp_31AssignVariableOp)assignvariableop_31_nadam_dense_54_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32³
AssignVariableOp_32AssignVariableOp+assignvariableop_32_nadam_dense_55_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33±
AssignVariableOp_33AssignVariableOp)assignvariableop_33_nadam_dense_55_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_339
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÊ
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_34½
Identity_35IdentityIdentity_34:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_35"#
identity_35Identity_35:output:0*
_input_shapes
: ::::::::::::::::::::::::::::::::::2$
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
¦ 
«
J__inference_sequential_27_layer_call_and_return_conditional_losses_4086930

inputs
conv1d_54_4086906
conv1d_54_4086908
conv1d_55_4086912
conv1d_55_4086914
dense_54_4086919
dense_54_4086921
dense_55_4086924
dense_55_4086926
identity¢!conv1d_54/StatefulPartitionedCall¢!conv1d_55/StatefulPartitionedCall¢ dense_54/StatefulPartitionedCall¢ dense_55/StatefulPartitionedCall 
!conv1d_54/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_54_4086906conv1d_54_4086908*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv1d_54_layer_call_and_return_conditional_losses_40867052#
!conv1d_54/StatefulPartitionedCall
 max_pooling1d_54/PartitionedCallPartitionedCall*conv1d_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling1d_54_layer_call_and_return_conditional_losses_40866632"
 max_pooling1d_54/PartitionedCallÃ
!conv1d_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_54/PartitionedCall:output:0conv1d_55_4086912conv1d_55_4086914*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv1d_55_layer_call_and_return_conditional_losses_40867392#
!conv1d_55/StatefulPartitionedCall
 max_pooling1d_55/PartitionedCallPartitionedCall*conv1d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling1d_55_layer_call_and_return_conditional_losses_40866782"
 max_pooling1d_55/PartitionedCallþ
flatten_27/PartitionedCallPartitionedCall)max_pooling1d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_27_layer_call_and_return_conditional_losses_40867622
flatten_27/PartitionedCall´
 dense_54/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_54_4086919dense_54_4086921*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_54_layer_call_and_return_conditional_losses_40867812"
 dense_54/StatefulPartitionedCallº
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_4086924dense_55_4086926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_55_layer_call_and_return_conditional_losses_40868082"
 dense_55/StatefulPartitionedCall
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0"^conv1d_54/StatefulPartitionedCall"^conv1d_55/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv1d_54/StatefulPartitionedCall!conv1d_54/StatefulPartitionedCall2F
!conv1d_55/StatefulPartitionedCall!conv1d_55/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
»
F__inference_conv1d_54_layer_call_and_return_conditional_losses_4087147

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        	               2
Pad/paddingsf
PadPadinputsPad/paddings:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
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
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:::S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¿
serving_default«
O
conv1d_54_input<
!serving_default_conv1d_54_input:0ÿÿÿÿÿÿÿÿÿ<
dense_550
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ìì
<
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
}_default_save_signature"ü8
_tf_keras_sequentialÝ8{"class_name": "Sequential", "name": "sequential_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_27", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_54_input"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_54", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [10]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_54", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_55", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_55", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_27", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_55", "trainable": true, "dtype": "float32", "units": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_27", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_54_input"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_54", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [10]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_54", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}, {"class_name": "Conv1D", "config": {"name": "conv1d_55", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_55", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_27", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_55", "trainable": true, "dtype": "float32", "units": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["mean_squared_error"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-08}}}}
ã


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*~&call_and_return_all_conditional_losses
__call__"¾	
_tf_keras_layer¤	{"class_name": "Conv1D", "name": "conv1d_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_54", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [10]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 1]}}
ý
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "MaxPooling1D", "name": "max_pooling1d_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_54", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ì	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Å
_tf_keras_layer«{"class_name": "Conv1D", "name": "conv1d_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_55", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 16]}}
ý
trainable_variables
	variables
 regularization_losses
!	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "MaxPooling1D", "name": "max_pooling1d_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_55", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ê
"trainable_variables
#	variables
$regularization_losses
%	keras_api
+&call_and_return_all_conditional_losses
__call__"Ù
_tf_keras_layer¿{"class_name": "Flatten", "name": "flatten_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_27", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ô

&kernel
'bias
(trainable_variables
)	variables
*regularization_losses
+	keras_api
+&call_and_return_all_conditional_losses
__call__"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_54", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
ò

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
+&call_and_return_all_conditional_losses
__call__"Ë
_tf_keras_layer±{"class_name": "Dense", "name": "dense_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_55", "trainable": true, "dtype": "float32", "units": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
÷
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
Ê
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
serving_default"
signature_map
&:$
2conv1d_54/kernel
:2conv1d_54/bias
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
­
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
°
Blayer_metrics
Cnon_trainable_variables
trainable_variables
	variables

Dlayers
Emetrics
regularization_losses
Flayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$2conv1d_55/kernel
:2conv1d_55/bias
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
°
Glayer_metrics
Hnon_trainable_variables
trainable_variables
	variables

Ilayers
Jmetrics
regularization_losses
Klayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Llayer_metrics
Mnon_trainable_variables
trainable_variables
	variables

Nlayers
Ometrics
 regularization_losses
Player_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Qlayer_metrics
Rnon_trainable_variables
"trainable_variables
#	variables

Slayers
Tmetrics
$regularization_losses
Ulayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_54/kernel
:
2dense_54/bias
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
°
Vlayer_metrics
Wnon_trainable_variables
(trainable_variables
)	variables

Xlayers
Ymetrics
*regularization_losses
Zlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_55/kernel
:2dense_55/bias
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
°
[layer_metrics
\non_trainable_variables
.trainable_variables
/	variables

]layers
^metrics
0regularization_losses
_layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
»
	btotal
	ccount
d	variables
e	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}

	ftotal
	gcount
h
_fn_kwargs
i	variables
j	keras_api"Ê
_tf_keras_metric¯{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}
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
2Nadam/conv1d_54/kernel/m
": 2Nadam/conv1d_54/bias/m
,:*2Nadam/conv1d_55/kernel/m
": 2Nadam/conv1d_55/bias/m
':%
2Nadam/dense_54/kernel/m
!:
2Nadam/dense_54/bias/m
':%
2Nadam/dense_55/kernel/m
!:2Nadam/dense_55/bias/m
,:*
2Nadam/conv1d_54/kernel/v
": 2Nadam/conv1d_54/bias/v
,:*2Nadam/conv1d_55/kernel/v
": 2Nadam/conv1d_55/bias/v
':%
2Nadam/dense_54/kernel/v
!:
2Nadam/dense_54/bias/v
':%
2Nadam/dense_55/kernel/v
!:2Nadam/dense_55/bias/v
ö2ó
J__inference_sequential_27_layer_call_and_return_conditional_losses_4086825
J__inference_sequential_27_layer_call_and_return_conditional_losses_4087034
J__inference_sequential_27_layer_call_and_return_conditional_losses_4086852
J__inference_sequential_27_layer_call_and_return_conditional_losses_4087088À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
/__inference_sequential_27_layer_call_fn_4087130
/__inference_sequential_27_layer_call_fn_4086901
/__inference_sequential_27_layer_call_fn_4087109
/__inference_sequential_27_layer_call_fn_4086949À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ì2é
"__inference__wrapped_model_4086654Â
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *2¢/
-*
conv1d_54_inputÿÿÿÿÿÿÿÿÿ
ð2í
F__inference_conv1d_54_layer_call_and_return_conditional_losses_4087147¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_conv1d_54_layer_call_fn_4087156¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥
M__inference_max_pooling1d_54_layer_call_and_return_conditional_losses_4086663Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
2__inference_max_pooling1d_54_layer_call_fn_4086669Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ð2í
F__inference_conv1d_55_layer_call_and_return_conditional_losses_4087173¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_conv1d_55_layer_call_fn_4087182¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥
M__inference_max_pooling1d_55_layer_call_and_return_conditional_losses_4086678Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
2__inference_max_pooling1d_55_layer_call_fn_4086684Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ñ2î
G__inference_flatten_27_layer_call_and_return_conditional_losses_4087188¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_flatten_27_layer_call_fn_4087193¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_54_layer_call_and_return_conditional_losses_4087204¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_dense_54_layer_call_fn_4087213¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_55_layer_call_and_return_conditional_losses_4087224¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_dense_55_layer_call_fn_4087233¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
<B:
%__inference_signature_wrapper_4086980conv1d_54_input£
"__inference__wrapped_model_4086654}&',-<¢9
2¢/
-*
conv1d_54_inputÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_55"
dense_55ÿÿÿÿÿÿÿÿÿ®
F__inference_conv1d_54_layer_call_and_return_conditional_losses_4087147d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_conv1d_54_layer_call_fn_4087156W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ®
F__inference_conv1d_55_layer_call_and_return_conditional_losses_4087173d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_conv1d_55_layer_call_fn_4087182W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_54_layer_call_and_return_conditional_losses_4087204\&'/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 }
*__inference_dense_54_layer_call_fn_4087213O&'/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
¥
E__inference_dense_55_layer_call_and_return_conditional_losses_4087224\,-/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_55_layer_call_fn_4087233O,-/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_flatten_27_layer_call_and_return_conditional_losses_4087188\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_flatten_27_layer_call_fn_4087193O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÖ
M__inference_max_pooling1d_54_layer_call_and_return_conditional_losses_4086663E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ­
2__inference_max_pooling1d_54_layer_call_fn_4086669wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÖ
M__inference_max_pooling1d_55_layer_call_and_return_conditional_losses_4086678E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ­
2__inference_max_pooling1d_55_layer_call_fn_4086684wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
J__inference_sequential_27_layer_call_and_return_conditional_losses_4086825w&',-D¢A
:¢7
-*
conv1d_54_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
J__inference_sequential_27_layer_call_and_return_conditional_losses_4086852w&',-D¢A
:¢7
-*
conv1d_54_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
J__inference_sequential_27_layer_call_and_return_conditional_losses_4087034n&',-;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
J__inference_sequential_27_layer_call_and_return_conditional_losses_4087088n&',-;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_sequential_27_layer_call_fn_4086901j&',-D¢A
:¢7
-*
conv1d_54_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_27_layer_call_fn_4086949j&',-D¢A
:¢7
-*
conv1d_54_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_27_layer_call_fn_4087109a&',-;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_27_layer_call_fn_4087130a&',-;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿº
%__inference_signature_wrapper_4086980&',-O¢L
¢ 
EªB
@
conv1d_54_input-*
conv1d_54_inputÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_55"
dense_55ÿÿÿÿÿÿÿÿÿ