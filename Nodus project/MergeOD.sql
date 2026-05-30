@@input1 := od_del3_road;
@@input2 := od_del3_iww;
@@input3 := od_del3_rail;
@@outputTable := od_del3;


drop table if exists @@outputTable;
drop table if exists pre;
create table pre as select * from @@input1 union all select * from  @@input2 union all select * from  @@input3;
create table  @@outputTable as select grp, org, dst, sum(qty) as qty from pre group by grp, org, dst;
drop table pre;

