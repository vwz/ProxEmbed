package dataPrepare.ProxEmbed;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Generate sub-paths by samplings.
 */
public class GenerateSubPathsFromSamplings {

	static String nodes_path=Config.NODES_PATH;
	static String conditional_random_walk_sampling_paths=Config.SAVE_PATH_FOR_RANDOMWALK_SAMPLINGS;
	static String truncated_type_name=Config.TRUNCATED_TYPE_NAME;
	static String subpaths_save_path=Config.SUBPATHS_SAVE_PATH;
	static int longest_length_for_window=Config.LONGEST_ANALYSE_LENGTH_FOR_SAMPLING;
	static int longest_lenght_for_subpaths=Config.LONGEST_LENGTH_FOR_SUBPATHS;
	
	public static void main(String[] args) {
		GenerateSubPathsFromSamplings g=new GenerateSubPathsFromSamplings();
		g.generateSubPathsFromSamplings(
				nodes_path, 
				conditional_random_walk_sampling_paths, 
				truncated_type_name, 
				subpaths_save_path, 
				longest_length_for_window,
				longest_lenght_for_subpaths);
	}

	/**
	 * Generate sub-paths by samplings.
	 */
	public void generateSubPathsFromSamplings(String nodesPath,String samplingsPath,String truncatedNodeType,String subPathsSavePath,int window_maxlen,int subpath_maxlen){
		Set<Integer> truncatedNodeIds=new HashSet<Integer>();
		Set<String> truncatedTypes=new HashSet<String>();
		String[] arr=truncatedNodeType.split(" ");
		truncatedTypes.addAll(Arrays.asList(arr));
		BufferedReader br=null;
		arr=null;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(nodesPath), "UTF-8"));
			String temp = null;
			while ((temp = br.readLine()) != null ) {
				temp=temp.trim();
				if(temp.length()>0){
					arr=temp.split("	");
					if(truncatedTypes.contains(arr[1])){
						truncatedNodeIds.add(Integer.parseInt(arr[0]));
					}
				}
			}
		} catch (Exception e2) {
			e2.printStackTrace();
		}
		finally{
			try {
				if(br!=null){
					br.close();
					br=null;
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		FileWriter writer =null;
		String t=null;
		List<Integer> path=new ArrayList<Integer>();
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(samplingsPath), "UTF-8"));
			writer = new FileWriter(subPathsSavePath);
			String temp = null;
			while ((temp = br.readLine()) != null ) {
				temp=temp.trim();
				if(temp.length()>0){
					path.clear();
					arr=temp.split(" ");
					for(String s:arr){
						path.add(Integer.parseInt(s));
					}
					t=analyseOnePath(path, truncatedNodeIds, window_maxlen, subpath_maxlen);
					if(t.length()>0){
						writer.write(t);
						writer.flush();
					}
				}
			}
		} catch (Exception e2) {
			e2.printStackTrace();
		}
		finally{
			try {
				if(writer!=null){
					writer.close();
					writer=null;
				}
				if(br!=null){
					br.close();
					br=null;
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * Generate sub-paths by one specific sampling path.
	 */
	private String analyseOnePath(List<Integer> path,Set<Integer> truncatedNodeIds,int maxWindowLen,int maxSubpathLen){
		StringBuilder sb=new StringBuilder();
		int start=0;
		int end=0;
		List<Integer> subpath=new ArrayList<Integer>();
		for(int i=0;i<path.size();i++){
			start=path.get(i);
			if(!truncatedNodeIds.contains(start)){
				continue;
			}
			for(int j=i+1;j<path.size();j++){
				end=path.get(j);
				if(!truncatedNodeIds.contains(end)){
					continue;
				}
				
				if(maxWindowLen>0 && (j-i)>maxWindowLen){
					break;
				}
				
				subpath.clear();
				for(int x=i;x<=j;x++){
					subpath.add(path.get(x)+0);
				}
				List<Integer> subpathNoRepeat=deleteRepeat(subpath);
				if(subpathNoRepeat.size()<2){
					subpathNoRepeat=null;
					continue;
				}
				
				if(maxSubpathLen>0 && subpathNoRepeat.size()>maxSubpathLen){
					continue;
				}
				
				sb.append(path.get(i)+"	"+path.get(j)+"	");
				for(int x=0;x<subpathNoRepeat.size();x++){
					sb.append(subpathNoRepeat.get(x)+" ");
				}
				sb.append("\r\n");
				subpathNoRepeat=null;
			}
		}
		return sb.toString();
	}
	
	/**
	 * Delete repeat segments for sub-paths
	 */
	public List<Integer> deleteRepeat(List<Integer> path){
		Map<Integer,Integer> map=new HashMap<Integer,Integer>();
		int node=0;
		List<Integer> result=new ArrayList<Integer>();
		int formerIndex=0;
		for(int i=0;i<path.size();i++){
			node=path.get(i);
			if(!map.containsKey(node)){
				map.put(node, i);
			}
			else{
				formerIndex=map.get(node);
				for(int j=formerIndex;j<i;j++){
					map.remove(path.get(j));
					path.set(j, -1);
				}
				map.put(node, i);
			}
		}
		for(int i=0;i<path.size();i++){
			if(path.get(i)!=-1){
				result.add(path.get(i));
			}
		}
		return result;
	}
}
