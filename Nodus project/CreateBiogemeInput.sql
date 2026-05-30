##################################################################
# Prepare wide input data for logit estimation
##################################################################

# Uncalibrated assignment
@@srcTable := path1_header;

# Output table
@@dstTable := biogeme_input_model1;

# Road OD table
@@od1Table := od_nuts2_road;

# IWW OD table
@@od2Table := od_nuts2_iww;

# Rail OD table
@@od3Table := od_nuts2_rail;


drop table if exists tmp2;
create table tmp2 as select grp, org, dst, qty, length, 
    ldcost+ulcost+trcost+tpcost+mvcost as cost, 
    ldcost+ulcost+trcost+tpcost as fcost, 
    mvcost as vcost, 
	ldduration+ulduration+trduration+tpduration+mvduration as duration,
	ldduration+ulduration+trduration+tpduration as fduration,
	mvduration as vduration,
	ldmode from @@srcTable;
create index tmp2idx on tmp2 (grp,org,dst,ldmode,cost);

# Create a table with the total quantity per mode, based on the minimum total cost
drop table if exists tmp;
create table tmp as select t.grp, t.org, t.dst, x.qty, t.length, t.cost, t.fcost, t.vcost, t.duration, t.fduration, t.vduration, t.ldmode 
from ( 
   select grp, org, dst, sum(qty) as qty, length, min(cost) as mincost, duration, ldmode
   from tmp2 group by grp, org, dst, ldmode
) as x inner join tmp2 as t on t.grp = x.grp and t.org=x.org and t.dst=x.dst and t.ldmode=x.ldmode and t.cost = x.mincost;


# Create wide table
create index tmpidx on tmp (grp,org,dst);

drop table if exists tmp2;
create table tmp2 as select org, dst, grp from tmp group by org,dst,grp order by org,dst,grp;
alter table tmp2 add cost1 DECIMAL(13,3);
alter table tmp2 add cost2 DECIMAL(13,3);
alter table tmp2 add cost3 DECIMAL(13,3);

alter table tmp2 add fcost1 DECIMAL(13,3);
alter table tmp2 add fcost2 DECIMAL(13,3);
alter table tmp2 add fcost3 DECIMAL(13,3);

alter table tmp2 add vcost1 DECIMAL(13,3);
alter table tmp2 add vcost2 DECIMAL(13,3);
alter table tmp2 add vcost3 DECIMAL(13,3);

alter table tmp2 add length1 DECIMAL(13,3);
alter table tmp2 add length2 DECIMAL(13,3);
alter table tmp2 add length3 DECIMAL(13,3);

alter table tmp2 add duration1 DECIMAL(13,3);
alter table tmp2 add duration2 DECIMAL(13,3);
alter table tmp2 add duration3 DECIMAL(13,3);

alter table tmp2 add fduration1 DECIMAL(13,3);
alter table tmp2 add fduration2 DECIMAL(13,3);
alter table tmp2 add fduration3 DECIMAL(13,3);

alter table tmp2 add vduration1 DECIMAL(13,3);
alter table tmp2 add vduration2 DECIMAL(13,3);
alter table tmp2 add vduration3 DECIMAL(13,3);

alter table tmp2 add qty1 DECIMAL(13,3) default 0;
alter table tmp2 add qty2 DECIMAL(13,3) default 0;
alter table tmp2 add qty3 DECIMAL(13,3) default 0;
create index tmp2idx on tmp2 (grp,org,dst);


# Make sure OD tables have one single entry for each grp,org,dst combination;
drop table if exists od1;
create table od1 select grp,org,dst, sum(qty) as qty from @@od1Table group by grp, org, dst;
drop table if exists od2;
create table od2 select grp,org,dst, sum(qty) as qty from @@od2Table group by grp, org, dst;
drop table if exists od3;
create table od3 select grp,org,dst, sum(qty) as qty from @@od3Table group by grp, org, dst;
create index od1idx on od1 (grp,org,dst);
create index od2idx on od2 (grp,org,dst);
create index od3idx on od3 (grp,org,dst);

# Update qty
update tmp2,od1 set tmp2.qty1 = od1.qty 
	where tmp2.grp=od1.grp 
	and   tmp2.org=od1.org 
	and   tmp2.dst = od1.dst; 
update tmp2,od2 set tmp2.qty2 = od2.qty
	where tmp2.grp=od2.grp 
	and   tmp2.org=od2.org 
	and   tmp2.dst = od2.dst;
update tmp2,od3 set tmp2.qty3 = od3.qty
	where tmp2.grp=od3.grp 
	and   tmp2.org=od3.org 
	and   tmp2.dst = od3.dst;

# Update costs
update tmp2,tmp set tmp2.cost1 = tmp.cost  
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=1;
update tmp2,tmp set tmp2.cost2 = tmp.cost  
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=2;
update tmp2,tmp set tmp2.cost3 = tmp.cost  
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=3;
	
# Update fcosts
update tmp2,tmp set tmp2.fcost1 = tmp.fcost  
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=1;
update tmp2,tmp set tmp2.fcost2 = tmp.fcost  
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=2;
update tmp2,tmp set tmp2.fcost3 = tmp.fcost  
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=3;

# Update vcosts
update tmp2,tmp set tmp2.vcost1 = tmp.vcost  
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=1;
update tmp2,tmp set tmp2.vcost2 = tmp.vcost  
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=2;
update tmp2,tmp set tmp2.vcost3 = tmp.vcost  
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=3;

# Update lengths
update tmp2,tmp set tmp2.length1 = tmp.length 
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=1;
update tmp2,tmp set tmp2.length2 = tmp.length 
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=2;
update tmp2,tmp set tmp2.length3 = tmp.length 
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=3;

# Update durations and convert them in hours
update tmp2,tmp set tmp2.duration1 = tmp.duration/3600
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=1;
update tmp2,tmp set tmp2.duration2 = tmp.duration/3600
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=2;
update tmp2,tmp set tmp2.duration3 = tmp.duration/3600
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=3;

# Update fdurations and convert them in hours
update tmp2,tmp set tmp2.fduration1 = tmp.fduration/3600
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=1;
update tmp2,tmp set tmp2.fduration2 = tmp.fduration/3600
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=2;
update tmp2,tmp set tmp2.fduration3 = tmp.fduration/3600
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=3;

# Update vdurations and convert them in hours
update tmp2,tmp set tmp2.vduration1 = tmp.vduration/3600
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=1;
update tmp2,tmp set tmp2.vduration2 = tmp.vduration/3600
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=2;
update tmp2,tmp set tmp2.vduration3 = tmp.vduration/3600
	where tmp2.grp=tmp.grp 
	and   tmp2.org=tmp.org 
	and   tmp2.dst = tmp.dst 
	and   tmp.ldmode=3;

# Clean temporary tables;
drop table if exists od1;
drop table if exists od2;
drop table if exists od3;
drop table if exists tmp;

# Remove records for which there is a qty but not path (must be fixed in input data)
delete from tmp2 where cost1 is null and qty2 > 0;
delete from tmp2 where cost2 is null and qty2 > 0;
delete from tmp2 where cost3 is null and qty3 > 0;

# Ignore cases where no acces by road
select grp, org, dst from tmp2 where cost1 is null;
delete from tmp2 where cost1 is null;

# Rename final table
drop table if exists @@dstTable;
rename table tmp2 to @@dstTable;

SELECT 'Done.' as '';
