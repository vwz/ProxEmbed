package dataPrepare.ProxEmbed;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Node class, contains id, type, in and out info.
 */
public class Node {

	private int id=-1;
	private String type=null;
	/**
	 * Index for type, not use now.
	 */
	private int typeId=-1;
	public List<Node> in_nodes=new ArrayList<Node>();
	public List<Node> out_nodes=new ArrayList<Node>();
	public List<Integer> in_ids=new ArrayList<Integer>();
	public List<Integer> out_ids=new ArrayList<Integer>();
	public Map<Node,List<List<Node>>> typePaths=new HashMap<Node, List<List<Node>>>();
	public Set<Node> neighbours=new HashSet<Node>(); 

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}
	
	public int getTypeId() {
		return typeId;
	}

	public void setTypeId(int typeId) {
		this.typeId = typeId;
	}

	@Override
	public int hashCode() {
		return this.id;
	}

	@Override
	public boolean equals(Object obj) {
		if(obj instanceof Node){
			Node node=(Node) obj;
			if(node.getId()==this.id){
				return true;
			}
		}
		return false;
	}

	@Override
	public String toString() {
		return "[id="+id+",neighbours=["+getNeighboursInfo()+"]]";
	}
	
	private String getNeighboursInfo(){
		StringBuilder sb=new StringBuilder();
		if(neighbours.size()==0){
			return "";
		}
		else{
			for(Node n:neighbours){
				sb.append(n.id+",");
			}
			return sb.toString();
		}
	}
}
